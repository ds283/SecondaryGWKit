# levin-phase-floor 07c — the chase below a refused leaf, and the Clenshaw–Curtis estimate

> **Part B, §1, §3 and §4** (the code, its tests and these records), written by the executing agent on the laptop, on
> the prompt as amended by its §9 (2026-11-19). **Part A, §2** (the measurement and its extension on R2-28, on the Mac
> Studio) is recorded in `docs/levin-phase-floor/07c/measurement/log.md`; prompt §9 holds the rulings. **Part C, §5
> and §6** (R32 and the census) runs later on the Mac Studio; the orchestrator appends its record.

**Subject:** prompt 07c §1, §3 and §4, as amended by §9 (Part B), on the laptop, in the main checkout
`/Users/ds283/Documents/Code/SecondaryGWKit`, branch `handover-remedial`.
- **Dispatch SHA:** `cff4237f` (pushed), the amendment at §2.7's stop. The parent module is
  `cff4237f:AdaptiveLevin/levin_quadrature.py` (SHA-256 `8956c7b6…`, 6,247 lines), 07b's code (`git diff --stat
  5e19c4b6 cff4237f -- AdaptiveLevin ComputeTargets LiouvilleGreen tools` is empty). The parent was run from a
  temporary detached worktree at `cff4237f` in the session scratchpad, the main checkout's `venv` symlinked into it;
  (S) from a second such worktree with Part A's variant (S) copied over its module. Both were removed afterwards.
- The reference implementation is (S)'s variant, `docs/levin-phase-floor/07c/measurement/scripts/variants/
  levin_quadrature_07c_S.py` (SHA-256 `09088683…`, checked before each use), with its diff `out/variants/S.diff`
  (`ef5a6e9e…`).
- `git log --oneline cff4237f..HEAD` was empty at the start and before staging.
- `runregistrykit list` at the start: 95 runs, 93 finished, 2 `unknown` (pre-registry), nothing `running` or stale. No
  Ray was started, no store was opened, no run was registered, nothing was written under `var/`, and nothing ran on
  the Mac Studio.

**Commit:** the code commit of this log (one commit, not pushed), on `cff4237f`.

**Result:**
- **§1 as amended is in `AdaptiveLevin/levin_quadrature.py`:** the stop below a refused leaf, (S)'s, under §1's
  names: `_LEVIN_CHASE_NOISE_FACTOR = 4.0` and `_LEVIN_CHASE_STOP_LIMIT = 6` with their basis, the module-level
  `_chase_at_noise_level(delta1, delta2, noise_yardstick)`, the three counts `num_chase_stops`,
  `num_chase_stopped_leaves` and `num_chase_noise_bisections`, the docstrings and comments of §1.5. The variant's
  scaffolding (`_VARIANT_07C`, `_MEASURE_07C`, `_E07C`, `_N07C`, `_S07C`) does not ship. **It equals (S) bit for
  bit**, (S)'s three counts equal to the commit's, on every call compared on the laptop: 29 test calls and
  4,320 LiouvilleGreen draw calls (Verification 3 and 4).
- **§3:** a new module `AdaptiveLevin/tests/test_levin_chase_stop.py`, 12 tests: (a) four, (b), (c) three, (d), (e),
  (f) two. (b)'s call is unconverged at the parent with all 6 of its depth-20 leaves below the refused leaf, and
  converged and honest at the commit (2.3e-5 of its abserr), the stop acting. **Every breakage caught**, (i)–(vii),
  each by the test §3 names for it.
- **Existing assertions changed:** two, in `test_levin_trust_yardstick` (test 2's key set admits the three counts) and
  `test_levin_roundoff_exit` (test 1: 9 phase-limited leaves become 11), each where the stop acts; no test goes from
  converged to unconverged or the reverse.
- **The seven roots:** AdaptiveLevin 129 → 141, the rise equal to the 12 tests added; ComputeTargets 781, CosmologyModels 39, LiouvilleGreen 156 (skipped 1), ObjectFactories 280, RunDrivers 127 and tools 64, each equal to its baseline.
- **The LiouvilleGreen draws:** no test newly failing (80 of 80 units pass under the parent, the commit and (S)); the 1,269 calls in which the stop does not act are bit-identical to the parent; it acts in 3,051 of 4,320 calls (4,819 roots stopped, 38,080 leaves); every call equals (S)'s bit for bit, counts included. **26 Levin calls converged at the parent are unconverged at the commit** (and under (S)), on their budget, no leaf untrusted, abserr 2.1–162 times the parent's and 1.008–10.7 times the request (Observations 5); none goes the other way.

## What shipped

### §1, the code (`AdaptiveLevin/levin_quadrature.py`, line numbers at the commit)

1. **§1.1, the noise level:** `_chase_at_noise_level(delta1, delta2, noise_yardstick) -> bool` (line 1463), module
   level, after `_leaf_rate_rule`: `delta1 <= C nu_B and delta2 <= C nu_B`, `C = _LEVIN_CHASE_NOISE_FACTOR`, in (S)'s
   arithmetic (`C * nu_B` formed for each comparison). In `_adaptive_levin`, `_leaf_at_chase_noise_level(seq)` (line
   5136) applies it to what the leaf's own trust check recorded; a leaf not yet checked is not at the noise level.
2. **§1.1, the root:** in `_check_leaf`, `refused_by_rate_rule` is set where 07b's `_leaf_rate_rule` refuses (line
   5050 and the refusal branch); every check records `chase_check[seq] = (delta1, delta2, noise_yardstick)`, and a
   refused leaf takes `chase_root.setdefault(seq, seq)` (lines 5069–5075), so the root is the outermost refused leaf.
   At each bisection both children take their parent's root (lines 5517–5522).
3. **§1.1, the count and the stop:** after `_add_leaf(left)`, `_add_leaf(right)` and the window's stop, a bisection of
   a leaf with a root, at the noise level, counts against the root; at `_LEVIN_CHASE_STOP_LIMIT` the root stops, once,
   and every current leaf with that root at the noise level is stopped, in the order of `leaves.values()` (lines
   5531–5550).
4. **§1.1, the reach:** at the end of `_check_leaf`, after `_push_candidate`, a leaf whose root has stopped and that is
   at the noise level is stopped (lines 5099–5104).
5. **§1.1, a stopped leaf:** `_stop_chase_leaf` (line 5144) leaves a floor-limited leaf as it is and otherwise calls
   `_floor_limit_stopped_window_leaf`, unchanged in its code, and counts the leaf.
6. **§1.2, the constants:** `_LEVIN_CHASE_NOISE_FACTOR = 4.0` and `_LEVIN_CHASE_STOP_LIMIT = 6` (lines 688–689), after
   `_LEVIN_LEAF_RATE_RANGE_MAX`, under one comment (lines 641–687) giving: what the chase is, as Part A measured it;
   C's basis (D1's 8,664 Clenshaw–Curtis bisections: within ν_B 15 %, 2ν_B 73 %, 4ν_B 88 %, 10ν_B 97 %; the 90th
   percentile 1.01 of δ₁/(φ_P + φ_L + φ_R)), its margin ((S2), C = 2, the same table 3), and where it does not reach
   (Levin leaves rarely at the noise level, 2,460 of 15,445; every chase call measured a group call; the analytic
   calls scored only by Part A's referenced ones); the limit's basis (dqage's iroff1, as the window stop) and that it
   was not measured; what a stopped leaf declares, measured at call level only, R2-28's carrying leaf not stopped,
   and on a Clenshaw–Curtis leaf the pair's understatement (D2: 5 of 14 resolved chase-stratum leaves) which the tree
   makes too; and (S)'s reach and cost on Part A's 454 calls.
7. **§1.3:** `converged` unchanged. A stopped leaf is floor-limited, not untrusted; its declaration counts against the
   budget as a stopped window's leaf does.
8. **§1.4, the counts:** `num_chase_stops`, `num_chase_stopped_leaves` and `num_chase_noise_bisections` (initialised at
   lines 4567–4572), ints, over the whole run, in the result dict after the refusal counts with a comment (lines
   5821–5827). Nothing else in the result changes.
9. **§1.5, the docstrings and comments rewritten:**
   - the module docstring's item on the trust check and the rate rules, a closing passage on the stop (lines 54–65);
   - the comment block above `_LEVIN_EDGE_RATE_RATIO_MAX`, its "Cost" bullet where it names the chase: the chase as it
     was, that 07c stops it, and R2-28 restated with 07c's numbers and its new issue (lines 627–634);
   - `_adaptive_levin`'s docstring where it said a call can be chased to `depth_max`: that the chase was, and four
     bullets on the stop (the root, the noise level, the count and the stop's reach, the stopped leaf's declaration),
     the counts, and the untouched call bit for bit (lines 4262–4287);
   - `_check_leaf`'s docstring, a paragraph on what each check records, the root and the stop's reach (lines
     4991–4999);
   - `_floor_limit_stopped_window_leaf`'s docstring, which now also stops a chase's leaf, and that a stopped leaf
     declares exactly what the tree declares for it at `depth_max` (lines 5155–5162);
   - `adaptive_levin_sincos`'s docstring, the three new counts (lines 6381–6393);
   - beyond §1.5's list: the result dict's comment beside the three counts (lines 5821–5824), and the comments on the
     new state, helpers and code (lines 4564–4566, 5066–5068, 5099–5100, 5126–5130, 5517–5518, 5531–5534).

### §3, the tests (`AdaptiveLevin/tests/test_levin_chase_stop.py`, new, 12 tests)

All without Ray or a datastore. The problems are θ = A e^{κx} on [0, 1.5], f = (e^{−x}, cos x), each sample multiplied
by 1 + η n(x), n(x) ∈ [−1, 1) a deterministic hash of the double's bits (`noise()` in the module: the same for a scalar
and an array element by element), η = 1e-12 where noise is used. The noise moves the integral by at most
η ∫(e^{−x} + |cos x|) = 1.774e-12, added to the true error in every honesty assertion on a noisy call. References:
mpmath's quad on the pieces between consecutive zeros of sin θ at 30 and 45 digits
(`docs/levin-phase-floor/07c/code/test_references_07c.py`, `out/test_references_07c.json`: agreement 4.5e-33, 1.1e-31
and 3.0e-32 relative). The parent's values are held as float hex, measured by running `cff4237f`'s module
(`compare_s_07c.py`); where a test needs the parent's call it runs the commit with `_LEVIN_CHASE_STOP_LIMIT` patched to
infinity (no root ever stops, and a count alone changes nothing), checked identical to the parent's module field by
field on all 29 calls compared (Verification 3), and asserts the parent's hex.

| §3 | test | what it asserts |
|---|---|---|
| (a) | `TestNoiseLevel` a1–a4 | C = 4; δ₁ = δ₂ = Cν_B at the noise level (and below it); either difference `nextafter(Cν_B, inf)` not; ν_B = 0: both differences 0 at the noise level, either the smallest positive double not; a NaN difference never |
| (b) | `TestTheChase.test_b_chase_below_a_refused_leaf` | below |
| (c) | `TestTheStopsReach` c1, c2, c3 | below |
| (d) | `TestOutermostRoot.test_d_the_outer_root_counts_and_stops` | below |
| (e) | `TestDeclarations.test_e_stopped_leaves_declare_as_at_depth_max` | below |
| (f) | `TestNotTouched` f1, f2 | below |

**(b), the chase.** A = 20, κ = 4, η = 1e-12, atol 1e-9, rtol 1e-7, the default order (16) and `depth_max` (20): θ′
ranges by e⁶ = 403 over the span, and the phase varies by 8,049 rad. The range rule refuses the span at depth 0 and
[0.75, 1.5] at depth 1; below them Levin leaves are refined until their phase varies by less than 6π, and the
Clenshaw–Curtis leaves there reach the noise level, where r is random and the residual |CC₄₇ − CC₂₄| carries the
integrand's noise above the floor.
- **At the parent:** 100 regions, **unconverged**, 6 depth-20 leaves, every one a Clenshaw–Curtis leaf returned
  untrusted (`num_untrusted_regions` 6), every one below the refused leaf [0, 1.5] (in [0.5491, 0.5492] and
  [0.58537, 0.58537]); value 1.62e-14 from the reference, 1.8e-5 of its abserr.
- **At the commit:** one root stopped after 6 bisections at the noise level, 20 leaves stopped; 26 regions, depth at
  most 8, **converged**, honest: true error 1.44e-14 against abserr 6.17e-10 (2.3e-5), no leaf returned untrusted.
- Asserted: converged, honest (with the noise bound as margin), refusals 2 range and 0 edge, one stop, stopped leaves
  positive, bisections counted equal to the limit, no untrusted region, depth below 20; the span refused by
  `_leaf_rate_rule` by the range rule (q > 100); the parent's hex reproduced on the parent's path, unconverged, no
  stop, 2 refusals, every depth-20 leaf Clenshaw–Curtis, untrusted and in the refused span, and every untrusted region
  one of them.

**(c), the stop's reach.** From the instrument (`trace_07c.py`, `out/trace_07c.txt`) on (b)'s call: the root stops at
the 17th bisection, the sixth at the noise level below it (the inner refused leaf's own bisection first, then five in
[0, 0.75]); the stop itself stops 11 leaves; 9 more are stopped by their own check after it, the two children of that
bisection and 7 leaves made later from Levin leaves of the subtree not at the noise level.
- **c1, made after the stop:** those 9 leaves, each asserted returned floor-limited (phase-limited, `trusted` None,
  value its own V_P), not at the cap, and at the noise level by `_chase_at_noise_level` with ν_B computed from the
  leaf's own solve and its halves' solved afresh as the driver solves them (`_trust_noise_yardstick`; the recomputed
  δ₁ asserted equal to the region's bit for bit); and the bisections counted equal to the limit times the stops: after
  the stop no leaf at the noise level below the root is bisected.
- **c2, not at the noise level:** six Levin leaves of the stopped subtree (δ₁/ν_B 12.9–183), each asserted not at the
  noise level by the same recomputed ν_B, returned trusted with its quarters' value, and equal to the same leaf in the
  parent's call field by field (depth, type, trusted, phase-limited, value, own value, value kind, residual, floor,
  δ₁, δ₂, r, declaration, e′).
- **c3, below a trust-test refusal:** A = 200, κ = 1, η = 1e-12, atol 1e-9, rtol 1e-7: θ′ ranges by e^1.5 = 4.5, so no
  rate-rule refusal; the instrument counts **18 bisections at the noise level below a leaf the trust test refused**
  (all below the root, which the trust test refuses). None is counted, none stopped: the three counts 0, the value and
  abserr the parent's hex, 32 regions, converged, honest (3.1e-5 of its abserr). Under breakage (vi) the stop acts
  there (Verification 2).

**(d), the outermost root.** On (b)'s call: two range refusals, [0.75, 1.5] refused by `_leaf_rate_rule` (q = e³), one
stop, the bisections counted equal to the limit; the two depth-2 leaves of [0.75, 1.5] returned stopped, and stopped
leaves in [0, 0.75] too, which no stop of the inner leaf's could reach. So the inner refused leaf's bisection counts
against the outer root, whose stop it is.

**(e), the declarations.** Three leaves of (b)'s call stopped at depth 8: [0.533203125, 0.5390625] untrusted with
r = 1.30; [0.521484375, 0.52734375] untrusted with r = 0.35 (δ₁ above its ν_B); [0.5390625, 0.544921875] trusted with
r = 0.04. Each returned with its own value V_P, declaring max(residual, δ₁ + δ₂), max(residual, δ₁)/(1 − r) and
max(residual, δ₁)/(1 − r) respectively. The parent's call at `depth_max` 8 (its hex asserted: 43 regions, unconverged)
returns the same three leaves at the cap: each declares there exactly what the commit's stopped leaf declares, bit for
bit, the untrusted two with the same V_P, the trusted one with its quarters' value. Measured on all 20 stopped leaves
of (b)'s call while choosing (scratch): 19 are at the parent's cap at their depth with the same declaration; the
twentieth ([0.4921875, 0.515625], depth 6) is not checked there.

**(f), not touched.** f1, 07b's (e1) (θ = 10³x on [0, 1], f = (e^{−x}, cos 30x), atol 1e-14): no refusal, the three
counts 0, the parent's hex, converged. f2, A = 2, κ = 4, no noise, atol 1e-14: two range refusals, one bisection
counted, no stop, no stopped leaf, the parent's hex, converged and honest (2.0e-4 of its abserr).

**The search for (b)** (§3: "If no call for (b) is found …, the agent stops and asks"), by scratch scripts loading the
parent's module and (S) side by side, on the family above with the deterministic noise, before the code was written:
- A ∈ {2, 20, 200}, κ = 4, η ∈ {0, 1e-12, 1e-11}, `depth_max` 10, atol 1e-14 and 1e-12: with η = 0 nothing chases;
  with noise the parent chases (unconverged, every capped leaf below the refused root, for A = 20 and 200), but at
  these atols (the request rtol|I|, 5.3e-10 for A = 20) (S) is unconverged too, on its budget: the stopped leaves'
  declarations carry CC₂₄'s truncation error on leaves near 6π (up to 2.6e-10);
- A ∈ {20, 200}, η ∈ {1e-12, 3e-12, 1e-11}, `depth_max` {10, 14}, atol {1e-9, 1e-8}: (S) converges only for A = 20,
  η = 1e-12;
- A = 20, η ∈ {3e-13, 5e-13, 1e-12, 2e-12}, `depth_max` {10, 20}, atol {1e-9, 3e-9, 1e-8}: η ≤ 5e-13 does not chase
  at the parent (converged); η = 1e-12 chases at the parent and converges under (S) at atol 1e-9 and 1e-8 with
  `depth_max` 20 (and at 1e-9, 1e-8 with 10); η = 2e-12 leaves (S) unconverged, with untrusted leaves at the cap whose
  differences exceed 4ν_B (Observations 2). (b) is A = 20, η = 1e-12, atol 1e-9, at the default `depth_max`.
- (c3) and the no-refusal calls came from a scan of the same family with κ ∈ {1, 4}, L ∈ {0.5, 1.5}, with and without
  `theta_deriv`, against a scratch copy of (S) with breakage (vi) applied.

### Existing assertions changed (§3, "Existing assertions")

Each is where the stop acts; every call involved equals (S) (`out/compare_s_07c.txt`).

1. **`test_levin_trust_yardstick.py`, `TestCodeIsTheVariants.test_bit_identical_to_the_variant`** (06m's test 2). The key
   set also admits the three counts (`CHASE_KEYS_07C`), left out of the compared records as 07b's are. The stop acts on
   si_1000_exp as it runs (one root, 6 leaves stopped, 12 bisections counted: 16 regions become 15, converged under
   both, the value bit-identical, abserr 1.479e-14 → 1.696e-14); the comparison there was already made with the rate
   rules refusing nothing (07b), where nothing is rooted. Asserted: the counts (1, 6, 12) on si_1000_exp as it runs, 0
   on the patched run and on every other cell (`CHASE_CELLS_07C`). Docstring paragraph added.
2. **`test_levin_roundoff_exit.py`, `TestRoundoffExitPremise.test_exit_against_its_premise_levin_branch`** (06b's test
   1, θ = 100eˣ on [0, 20], 07b's assertions on the code as it runs). The stop acts: one root, 7 leaves stopped (among
   them the range-refused [0, 2.5] at depth 3, at the noise level, in place of 07b's two depth-4 leaves), 12 bisections
   counted. `num_phase_limited_regions` **9 → 11** (7 by the stop, 4 by the round-off exit); the three counts asserted.
   Everything else stands: converged, 9 range refusals, every phase-limited leaf at depth 1 or more, honest (true error
   4.5e-16 against abserr 1.18e-14; 07b's 5.0e-16 against 9.66e-15) and within rtol. Docstring paragraph added.

No assertion goes from converged to unconverged or the reverse. **Where the stop acts in existing tests without changing
an assertion** (found by running the AdaptiveLevin root with a recording wrapper on `_adaptive_levin`, scratch):
`test_levin_leaf_values` test 1 on si_1000_exp, `test_levin_global_budget` test 3 (`test_inflated_floor_is_chased`,
the same problem) and test 8 (`test_determinism`, test 3's problem twice): each passes as written (converged, honest,
within rtol, deterministic).

### Records (`docs/levin-phase-floor/07c/code/`)

- `test_references_07c.py`, `out/test_references_07c.json`: the references of (b), (c3) and (f2).
- `compare_s_07c.py`, `out/compare_s_07c.{txt,json}`: the commit against (S) and the parent on 29 calls, the parent's
  path (limit infinite) against the parent's module, and the parent's values the tests quote.
- `trace_07c.py`, `out/trace_07c.txt`: the instrument on (b)'s and (c3)'s calls.
- `breakages_07c.py`, `out/breakages_summary.txt`, `out/breakages/`.
- `lg_draws_07c.py` (07b's harness copied, changed where marked "prompt 07c": the three counts recorded under the
  commit's or (S)'s names and left out of the result digest; three trees, the parent, the commit and (S); the report
  by whether the stop acts), `out/parent/`, `out/commit/`, `out/S/`, `out/lg_draws_report.{txt,json}`,
  `out/lg_run_stdout.txt`.
- Each script is `black`-formatted before its recorded run, and its outputs record its SHA-256; no `CLAUDE.md` change
  is needed.

## Predictions and verdicts

### Part B's acceptance (§7 item 2)

| Rule | Number | Verdict |
|---|---|---|
| §1 as amended | §1.1–§1.5; equal to (S) with equal counts on 29 test calls and 4,320 draw calls | holds |
| every test of §3 passing | the seven roots OK (Verification 1) | holds |
| every breakage caught | (i)–(vii), each by the test §3 names (Verification 2) | holds |
| every changed existing assertion listed | two AdaptiveLevin tests (above) | holds |
| the seven roots rising only by the tests added | AdaptiveLevin +12 = tests added; the other six equal | holds |
| the LiouvilleGreen draws with no new failure | 0 of 80 units newly failing; 1,269 of 1,269 untouched calls bit-identical; 26 calls converged → unconverged (reported, Observations 5) | holds |
| §3 (b) found unconverged by a chase at the parent, converged and honest at the commit | 6 of 6 depth-20 leaves below the refused leaf at the parent; 2.3e-5 of its abserr at the commit | holds |

### §5's predictions (R32), copied before they run — not yet scored (Part C)

As 07b §5, with records under `var/levin-phase-floor-07c/rehearsal/`, slug `r32-07c`, `--prompt 07c`,
`--job-name levin-phase-floor-07c`, and the fingerprint `1b332ffa…`. The parent rehearsal is **R31**
(`levin-phase-floor-07b-r31-07b-20261116T191849`, `var/datastores/levin-phase-floor-07b-r31*`, kept),
built at `0a103b5a`, whose code is `5e19c4b6`'s.
1. **S0′ — R31 is the parent.** `git diff --stat 5e19c4b6 <code commit's parent> -- . ':!docs'
   ':!prompts'` lists nothing the build imports. **Holds** if so; if not, it stops and asks.
2. **The lock:** `acquire`, `set-pid` at the launch, `release` after the readers, each a shell command
   of its own.
3. **R32 against R31,** by `tools.store_value_diff` with `--json` and 05's `classify_non_qsi.py` (job
   names normalised in a copy): outside `QuadSourceIntegral` every value table identical, every other
   difference of the four volatile kinds, no row in one store only; inside it, every moved row
   tabulated, with the rows whose `total_converged` changes in each direction.
4. **S3 — consistency.** Every row with finite `total_abserr` in both has
   $|{\rm total}_{32} - {\rm total}_{31}| \le {\rm total\_abserr}_{31} + {\rm total\_abserr}_{32}$, or is
   one §9 predicts to move further, listed.
5. **S1, S2 and S4** re-scored on R32 as 07b §5 item 5.
6. **The readers** run on R32, exit 0, and the store is byte-identical before and after. **The lock** is
   released and `ray-processes` lists nothing. **R31 and R32 are kept.**

### §6's predictions (the census), copied before they run — not yet scored (Part C)

07b's census harness (`docs/levin-phase-floor/07b/census/scripts/`), copied, changed where marked
"prompt 07c", re-run at the code commit against 07b's census (`levin-phase-floor-07b-census-20261117T010200`,
checkpoint `396784bc…`) as the "before". Diagnostics as §9 sets them, recording on every call whether
the fix acted and where.

**Gate Y0, before the run:**
- **The parent's code is 07b's:** as S0′.
- **The harness reproduces 07b's census at the parent,** bit for bit, on 06k's 100 K0 units. One
  registered run, slug `k0`.
- **The instrument changes nothing:** a unit where the fix acts and one where it does not, at the new
  code, with and without it, bit-identical.
- **The new code is deterministic:** two runs of one unit where the fix acts, bit-identical.
- **The projection** from K0 at the new code; beyond 48 hours is a stop, with what would be cut.

**Predictions, fixed before the run** (README §5 rule 3), copied into `state/predictions.md` unchanged,
with §9's additions:
- **Y1 — nothing else moves.** Every call in which the fix does not act (§9 names the counts) equals
  07b's call bit for bit in value, abserr, `converged`, `num_regions` and maximum depth. Every row all
  of whose calls are such equals 07b's in `total` and `total_abserr`. **Holds** if true at both rtols.
- **Y2 — no Levin leaf is returned across a stationary point** (07a's W2).
- **Y3 — every phase with something to locate is located** (07a's W3).
- **Y4 — honest where it is referenced.** Every call referenced in 07b (U4's 190) and in this prompt's
  Part A (R4, R5, R2-28), scored from the census against its reference by 07's rule. **Holds** if every
  one is honest, or unconverged under §9's rule.
- **Y5 — the gate is consistent, and the guard is a backstop** (07a's W5, on gated solves).
- **Y6 — 06b's 24 are honest,** against $S_{10}$ by 06b's rule, or unconverged.
- **Y7 — the pairs:** $r_g$ over all 2,484 pairs; every pair has $r_g \le 1$, or a member unconverged.
- **Y8 — the code is the chosen variant:** every census call Part A replayed equals the variant's
  replay record bit for bit in value, abserr, `converged`, regions and maximum depth.

**Descriptive, not scored:** where the fix acts, by class, band, $k$ and rtol; `converged` changes
against 07b's and against 07a's, in each direction, and the calls still unconverged with every
depth-20 leaf below a refused leaf; rows at depth 20; the abserr ratio against 07b's; values moved past
the sum of the two codes' abserrs (**the sum**, not each separately); regions and cost against 07b's.

### §9.5's additions to §5's and §6's predictions (Y9 among them), copied before they run — not yet scored (Part C)

- **Y1 counts** a call untouched if `num_chase_stops` is 0, and compares value, abserr, `converged`,
  `num_regions` and the maximum depth. Every call with no rate-rule refusal is untouched (07b's U1:
  60,160 calls); on Part A's 454, 126 are, and on R2's 60 random calls, 47.
- **Y4, on the user's ruling.** R2-28 is predicted converged and dishonest at 1.143 of its abserr at
  both rtols (its census call `c-1e-7-0-13757-5` and `c-1e-8-0-13757-5`). Its error is the carrying
  leaf's (§9.2), not the stop's, and belongs to the issue opened in §9.6. **Y4 holds** if every other
  referenced call is honest or unconverged; R2-28 is reported with its ratio against that issue. Any
  other referenced call converged and dishonest fails Y4.
- **Y6 and Y7 are predicted to hold.** 06b's W2 calls and the two pairs that Part A replayed are honest
  under S. The rest were not replayed; the prediction rests on S leaving no other referenced call
  converged and dishonest.
- **Y8 — the code is (S).** Every census call that Part A replayed (§2.1's, both rtols) equals (S)'s
  replay record bit for bit in value, abserr, `converged`, regions and maximum depth. **Holds** if all
  do.
- **Y9 — the chase ends.** No call is unconverged with every depth-20 leaf below a leaf a rate rule
  refused. Of 07b's 198 newly unconverged calls, exactly five stay unconverged: `c-1e-8-1-2229-4`,
  `c-1e-8-3-13490-2`, `c-1e-8-3-13490-3`, `c-1e-8-3-13615-2` and `c-1e-8-3-13615-3`, each untouched
  and equal to 07b's call. **Holds** if both are true. Y9 is the header's closing condition for
  `[07b-rate-range-rule-chases-noise-leaves-beyond-the-window-stop]`, and is scored with Y1–Y8 (§7).
- **S3 (§5).** R31's record (`G+q+r` at $(10^5, 10^6, 10^6)$, $z = 144627.7$) is predicted to converge
  at 86 regions, honest at 0.053 of its abserr. Its row is named as predicted to move, with
  `total_converged` possibly false → true. Any other row beyond S3's bound is reported with the stops
  in its calls.
- **Descriptive, added:** the stops by class, band, $k$ and rtol; by the harness's instrument (marked
  "prompt 07c"), each stopped leaf's route (Clenshaw–Curtis or Levin), depth and declaration, and its
  share of its call's abserr; `converged` changes against 07b's in each direction, with the calls
  newly unconverged listed (none is expected).

**Part B's review** replays from copies of their captures at the code commit:
- R2-28, equal to (S)'s replay record in `float.hex` (value `-0x1.654fbbd987ff2p-96`, abserr
  `0x1.11de231bf47dap-122`, 100 regions, depth 13, converged);
- one other chase call, `c-1e-8-1-2113-1`, equal to (S)'s record;
- `c-1e-8-1-2229-4` (untouched), equal to (S)'s record and to 07b's census.

## Deviations from the prompt

1. **`IMPLEMENTATION CHOICE`** — the names §1 leaves to the agent: the module-level function `_chase_at_noise_level`;
   in `_adaptive_levin`, `refused_by_rate_rule`, `chase_root`, `chase_check`, `chase_counts`, `chase_stopped`,
   `_leaf_at_chase_noise_level` and `_stop_chase_leaf`. The function compares as (S) does, `C * nu_B` formed for each
   comparison, so the bits are (S)'s.
2. **`IMPLEMENTATION CHOICE`** — the test module's name, `test_levin_chase_stop.py`, and its parts: (a) as four tests,
   (c) as three, (f) as two; 12 in all.
3. **`IMPLEMENTATION CHOICE`** — (b) perturbs the integrand by a deterministic hash noise of relative size 1e-12, and
   runs at atol 1e-9, about 1.9e-7 of |I| (the request max(atol, rtol|I|) is atol's). At the census's request, rtol|I|
   alone, the stopped leaves' declarations on this problem, which carry CC₂₄'s truncation error on leaves near 6π, do
   not meet the budget (the search above); §3 allows the noise, and the depth cap is the default, not a shallow one.
4. **`IMPLEMENTATION CHOICE`** — the tests reproduce the parent by patching `_LEVIN_CHASE_STOP_LIMIT` to infinity,
   asserting the parent's hex (measured by running `cff4237f`'s module), rather than running the parent; the patched
   path is checked identical to the parent's module, field by field, on all 29 calls of `compare_s_07c.py`.
5. **`IMPLEMENTATION CHOICE`** — (c) names its leaves from an instrument on the commit's module (`trace_07c.py`), and
   asserts them through the result's regions and counts, as §3 asks; to show a leaf at the noise level or not, (c1) and
   (c2) also solve the leaf and its halves afresh for ν_B, which the result does not carry, with the recomputed δ₁
   asserted equal to the region's bit for bit.
6. **`IMPLEMENTATION CHOICE`** — (c) item 3 and (f) item 1 are distinct calls: (c3) is a noisy call with no rate-rule
   refusal in which the instrument counts 18 bisections at the noise level below trust-test refusals; (f1) is 07b's
   (e1), with no noise. (c3) is also a call with no rate-rule refusal, bit-identical to the parent.
7. **`IMPLEMENTATION CHOICE`** — breakage (vii), "the trusted declaration", is max(residual, δ₁)/(1 − r) where r < 1
   (equal there to the untrusted declaration, so (vii) changes the value only) and max(residual, δ₁) where r ≥ 1, where
   the trusted form has no meaning; with the quarters' value in both cases.
8. **`IMPLEMENTATION CHOICE`** — §1.5's comment block above `_LEVIN_EDGE_RATE_RATIO_MAX` keeps its sentence on R2-28
   (07b's 1.107), restated with 07c's numbers (converged at 1.143 under the stop) and the issue opened at §9.6, and
   points to the new constants' comment.
9. **`IMPLEMENTATION CHOICE`** — comments beyond §1.5's list: the result dict's comment beside the three counts, and the
   comments on the new state, helpers and code in `_adaptive_levin` (listed above, item 9). §1 item 6 asks for every
   comment describing the stop.
10. **`IMPLEMENTATION CHOICE`** — records beside the draws' harness in `docs/levin-phase-floor/07c/code/`: the
    references, the comparison with (S) and the parent, the instrument and the breakages.
11. **`IMPLEMENTATION CHOICE`** — the draws run a third tree, (S), and compare the commit with it call by call, counts
    included (§9.5 Y8 on the laptop's problems); 07a's route instrument and 07b's refusal counts are kept unchanged.
12. **`IMPLEMENTATION CHOICE`** — the breakages ran before the commit, on the working tree: after each, the module's
    SHA-256 and `git status --porcelain -- AdaptiveLevin LiouvilleGreen` are checked equal to their values before the
    first (the tree was not clean then, holding this commit's changes).

## Verification performed

1. **The seven roots at the commit, serially** (the code commit's working tree; the SHA-256 of `levin_quadrature.py`,
   `d141a80b…`, and of every `AdaptiveLevin/tests/` module checked equal before and after):

| root | baseline (dispatch SHA) | at the commit |
|---|---|---|
| AdaptiveLevin | 129 OK | 141 OK (+12, the tests added) |
| ComputeTargets | 781 OK | 781 OK (the wall-clock flake did not fire) |
| CosmologyModels | 39 OK | 39 OK |
| LiouvilleGreen | 156 OK, skipped 1 | 156 OK, skipped 1 |
| ObjectFactories | 280 OK | 280 OK |
| RunDrivers | 127 OK | 127 OK |
| tools | 64 OK | 64 OK |

   Only the log and the board changed after that.

2. **The breakages** (`breakages_07c.py`, `out/breakages_summary.txt`), each applied as text replacements, the new
   module run, the module restored (SHA-256 back to `d141a80b…` and `git status --porcelain -- AdaptiveLevin
   LiouvilleGreen` as before the first, after each):

| breakage | caught by (tests failing) | named in §3 |
|---|---|---|
| (i) `_LEVIN_CHASE_STOP_LIMIT` = inf | (b), (c1); also (d), (e) | (b) and (c) |
| (ii) `_LEVIN_CHASE_NOISE_FACTOR` = inf | (a) a1, a2, a3; (c1), (c2); also (b), (d), (e) | (a) and (c) |
| (iii) the stop not reaching leaves made after it | (c1); also (b), (d), (e) | (c) |
| (iv) children not taking their parent's root | (b), (c1); also (d), (e), (f2) | (b) or (c) |
| (v) the innermost refused leaf as the root | (d); also (b) | (d) |
| (vi) bisections counted below any untrusted leaf | (c3) | (c) or (f) |
| (vii) a stopped untrusted leaf keeping its quarters' value and the trusted declaration | (e); also (c1), (d) | (e) |

   A first run of the breakages, before a comment-only change to the module (`8391eab5…` → `d141a80b…`), failed the
   same tests in every case; the recorded run is the second.

3. **The comparison with (S):** `compare_s_07c.py` runs the commit, (S) (SHA-256 checked) and the parent on 29 calls
   (the new module's five calls, 07b's five rate-rule calls, the nine yardstick cells, the seven leaf-value cells, the
   round-off exit's test 1 and the global budget's tests 1 and 3): 29 of 29 identical to (S) in every result key but
   `elapsed` and the counts and in every region field by field, with (S)'s three counts equal to the commit's; the stop
   acts on 6 ((b) at both depth caps, si_1000_exp three times, θ = 100eˣ); the other 23 identical to the parent; the
   parent's path (`_LEVIN_CHASE_STOP_LIMIT` infinite) identical to the parent's module on all 29. The draws: 4,320 of
   4,320 Levin calls identical to (S)'s run (value, abserr, every region field, every result key but `elapsed`,
   `location_skipped_phases` and the counts) with the three counts equal, and 80 of 80 units pass under (S). §9.5's
   review replays (R2-28, `c-1e-8-1-2113-1`, `c-1e-8-1-2229-4`) need captures that are on the Mac Studio, and are the
   orchestrator's.

4. **The LiouvilleGreen draws:** `lg_draws_07c.py` over seeds 0–19, the four default tests of
   `test_3bessel_analytic.py`, 80 units under each tree (`out/lg_draws_report.txt`):
   - test outcomes, parent → commit and parent → (S): pass → pass 80, newly failing 0;
   - 4,320 Levin calls paired, none unpaired; the rate rules refuse in 3,811 and a bisection is counted against a root
     in 3,709. **The stop does not act in 1,269: all bit-identical to the parent.** It acts in 3,051 (4,819 roots
     stopped, 38,080 leaves stopped, 43,940 bisections counted): every one moves; `converged` (parent, commit): (True,
     True) 3,025, **(True, False) 26**, none the other way; the largest move 4.4e-4 of the parent's abserr; regions
     147,029 → 140,896 (where it acts, 109,166 → 103,033); no call at depth 20 under either; wall summed over units
     1,220.9 s → 1,177.3 s ((S) 1,183.5 s);
   - the three-Bessel results (|value − analytic| ≤ abserr), by whether the stop acts in any of their calls: acted 1,018
     results, honest 587 → 595, converged 877 → 851, dishonest and converged 431 → 419; untouched 62, unchanged (honest
     59, dishonest and converged 3).

5. **The instrument** (`trace_07c.py`): on (b)'s and (c3)'s calls the instrumented module's call equals the commit's
   (value, abserr, `converged`, regions, the counts, each region's span, value and declaration) before its log is
   written.

6. **The references** (`test_references_07c.py`): (b) 4.5e-33, (c3) 1.1e-31 and (f2) 3.0e-32 relative agreement
   between the 30- and 45-digit readings.

7. **black** on every Python file changed or added; `git diff --stat` and `git status` list only the dispatch's paths.
   `git log --oneline cff4237f..HEAD` empty before staging.

## Observations not acted on

1. **The noise-level chase is not confined to refused subtrees on the synthetic family.** With the hash noise, calls
   with no rate-rule refusal also reach depth 20 unconverged at the parent and at the commit, where the stop does not
   act by design: θ = 20e^{4x} on [0, 0.5] (η = 1e-12, 1,188 regions), θ = 200eˣ on [0, 1.5] (η = 3e-12, 257 regions),
   and the same with no `theta_deriv` (no rates, so no rule). The trust check's r is random at the noise level wherever
   a leaf gets there and is untrusted; below refused leaves the rules drive leaves there whatever the budget. The census
   measured its chase below refused leaves (192 of 198); whether production calls with no refusal chase this way was not
   measured here. Not acted on: outside §1.
2. **Integrand noise above 4ν_B escapes both exits.** On (b)'s problem at η = 2e-12 the commit, like (S), stays
   unconverged with untrusted Clenshaw–Curtis leaves at the cap whose differences exceed 4ν_B: their residual is above
   their floor, so the round-off exit is not considered, and they are not at the noise level, so the stop does not take
   them. The window in noise amplitude where the stop ends the synthetic chase is narrow (1e-12 yes, 2e-12 no). The
   census's chase leaves sit at δ₁/ν_B medians 0.65–0.96 (§9.2), inside the stop's reach; this is Part A's proposed
   `[07c-integrand-noise-above-the-cc-floor]` (its `issues.md` item 1), put to the user at the records commit.
3. **A refused Levin leaf at the noise level is itself stopped.** On `test_levin_roundoff_exit` test 1 the range rule
   refuses [0, 2.5] (q = e^{2.5} = 12.2), whose check finds it at the noise level, and the stop returns it with V_P and
   max(residual, δ₁)/(1 − r); on (b) the two depth-2 Levin leaves of [0.75, 1.5] likewise. This is (S) as specified (a
   leaf at the noise level is stopped whatever its route; §0.3's lesson concerns leaves not at the noise level), and
   both calls are honest; recorded because a range-refused leaf returned on its r-based declaration is what the rule
   refuses elsewhere.
4. **Stopped Clenshaw–Curtis leaves near 6π declare CC₂₄'s truncation error.** On (b), [0.4921875, 0.515625] and
   [0.66796875, 0.6796875] declare 2.1e-10 and 2.6e-10 (their residual |CC₄₇ − CC₂₄|), while the call's true error is
   1.4e-14: an overstatement, `[06d-cc-nested-estimate-measures-the-coarse-rule]`'s, which the tree makes wherever the
   fallback declares. It is why (b) needs atol 1e-9 (Deviation 3).
5. **The stop makes 26 LiouvilleGreen draw calls unconverged that converge at the parent** (`out/lg_draws_report.txt`
   §2; the same under (S)). Each is unconverged on its budget, with no leaf returned untrusted and no depth-20 leaf
   (its maximum depth 9–16); its abserr is 2.1–162 times the parent's and 1.008–10.7 times its request (rtol 1e-8 or
   1e-10); its value moves by at most 8.3e-5 of the parent's abserr. The likely cause is the stopped leaves'
   declarations (a stopped untrusted leaf declares max(residual, δ₁)/(1 − r) or max(residual, δ₁ + δ₂), and a stopped
   Clenshaw–Curtis leaf near 6π its CC₂₄ error) where refinement at the parent went on to converge; it was not
   attributed leaf by leaf. Of the 26 three-Bessel results these calls feed, all converged at the parent: 14 are honest
   under both codes, 8 dishonest at the parent are honest at the commit (0.33–0.98 of their abserr), and 4 are
   dishonest under both (571–1.2e4 of their abserr; 07b's population of results its rules do not reach). Overall the
   results where the stop acts go from 587 to 595 honest and from 431 to 419 converged and dishonest. §9.5 expects no
   census call newly unconverged; these calls suggest the census may show some at rtol 1e-8. Not a §8 stop (the code
   equals (S); no test fails). Recorded for the orchestrator and Part C's scoring.

## State handed on

- The code commit (not pushed). Part C (§5 R32, §6 the census) runs on the Mac Studio after it is pushed; §9.5's
  replays of R2-28, `c-1e-8-1-2113-1` and `c-1e-8-1-2229-4` are the orchestrator's review.
- No issue moves in this commit (§4); `docs/OPEN_ISSUES.md` is not touched.
- The scratch worktrees are removed and pruned; nothing is running.

## Part C: the Mac Studio record (appended by the orchestrator, 2026-11-20)

Part C ran from 2026-11-19 to 2026-11-20 on the Mac Studio (`mps026298`), through an orchestrator session there, from
the worktree `SecondaryGWKit-lpf07c-cc3368c` at Part C's dispatch SHA `cc3368ce`, whose code is the code commit
`67b5af74`'s (the orchestrator note's §C.5). The Mac Studio orchestrator dispatched a rehearsal agent (R32), then a
census agent, and held them at the note's planned stops 1, 2 and 3. It checked each stop with readers of its own, then
reviewed Part C in the note's §C.2 checks, **all passing**. The records are in `docs/levin-phase-floor/07c/`:
- `rehearsal/` and `census/`: each agent's log, verification, issues, scripts and outputs. The two logs are appended
  below as written on the Mac Studio;
- `review/`: the Mac Studio orchestrator's stop records, scripts and hashes, and its review (`review/review.md`).

What stays on the Mac Studio is in `07c/KEPT_ON_MAC_STUDIO.md`, Part C. `VERIFICATION.md` §7c is the summary. Part A's
log (`07c/measurement/log.md`) follows the census agent's, in full, with its headings moved down one level to sit under
this log's own.

**The verdicts on the predictions above:**

| | Verdict | Numbers |
|---|---|---|
| Part B's review (§9.5) | **holds** | R2-28, `c-1e-8-1-2113-1` and `c-1e-8-1-2229-4` replayed at `67b5af74` from copies of their captures, each equal to (S)'s replay record leaf by leaf (Decisions, **07c Part B lands**) |
| §5, S0′ | **holds** | `git diff --stat 5e19c4b6 cff4237f` lists `CLAUDE.md` alone; `levin_quadrature.py` at `cff4237f` is `5e19c4b6`'s (`8956c7b6…`); R31 is the parent |
| §5, the fingerprint | **holds** | R32 `1b332ffa…`, re-taken without `--write`; R32's files hash the same before and after |
| §5, the lock | **holds** | acquired, `set-pid` at the launch, released after the readers at 14:49:20Z; one `acquire` shared a zsh command with an exited heredoc (rehearsal deviation 4, harmless) |
| §5, R32 against R31 | **holds, and tabulated** | outside `QuadSourceIntegral` 360,462 rows identical, the rest timing, `label`, `GkSource.parents_digest` (291) and rho-reuse metadata (26); inside it 305 of 864 rows moved, the largest 1.05×10⁻⁵ of R31's abserr; `total_converged` 1 up (861 → 862), 0 down; Levin-record regions 219,776 → 211,594 |
| §5, S3 | **holds** | 0 violations in 864 rows (251 with a moved total), scored exactly; the largest 5.25×10⁻⁶. §9.5's named record (`G+q+r`, $(10^5, 10^6, 10^6)$, $z = 144627.7$) converged at 86 regions (R31: 143, unconverged), honest at 0.0533 against Part A's `ref-R31-3-20363-g1`, equal to (S)'s replay bit for bit |
| §5, S1, S2, S4 | **hold** | S1 12 of 12 converged, located and honest, worst 0.0670 (w06); w08–w11 moved, one chase stop each. S2 0.0202 and 0.111. S4 4 of 4 located, guarding nothing, honest at 4.4×10⁻⁵–6.2×10⁻⁴, equal to R31's |
| §5, the replay and the readers | **hold** | 445 of 445 replay records reproduce R32; 6 of 6 readers exit 0, R32 unchanged; R31 and R32 kept |
| §6, Gate Y0 | **holds** | the parent's code is 07b's; 100 of 100 K0 units (2,110 calls) equal 07b's census; the instrument changes nothing (`h-1e-8-0-33929`, `h-1e-8-3-14505`); the code is deterministic |
| §9.4, the projection | **holds** | central 43,312 s, 51,833 s with 07b's shortfall of 1.197, within 48 h; measured 43,080 s |
| §6, Y1 | **holds** | 207,310 untouched calls equal 07b's in hex value, abserr, `converged`, `num_regions` and maximum depth; 8,508 units untouched in every call equal in `total` and `total_abserr` |
| §6, Y2 | **holds** | 0 returned Levin leaves hold a sign change, 182,066 calls scanned |
| §6, Y3 | **holds** | 0 violations; 784 skipped phases, all `zero_rate` |
| §6 and §9.5, Y4 | **holds, as §9.5 predicted** | 312 scorings on 306 calls (07b's U4 190; Part A's R4 60, R5 60, R2-28 at both rtols). Converged and dishonest: only R2-28, at 1.1434 at both rtols, reported against `[07c-r2-28-trusted-levin-leaf-hides-a-common-error]`. Unconverged: v04, `c-1e-8-3-13490-2` and `c-1e-8-3-13615-3`. Largest other converged ratio 0.805 (R2-08~o). R4 58 of 60 converged (07b 2 of 60), its largest 0.416; R5 60 of 60 honest. Six reported, not scored (R2-27, R2-29, R3a-0-33666-a13, each at both rtols; all honest) |
| §6, Y5 | **holds** | 0, 0 and 0 on gated solves; 18,238 forced node-partition sums in 4,416 calls (= 07b) |
| §6 and §9.5, Y6 | **holds, as predicted** | 24 of 24 converged and honest against $S_{10}$, at $1.6\times10^{-5}$–0.828 |
| §6 and §9.5, Y7 | **holds, as predicted** | $r_g \le 1$ on all 2,484 pairs, the largest 0.467 (`3-34923` #1, = 07b) |
| §9.5, Y8 | **holds** | 453 of 453 ids equal (S)'s replay records in all five outputs |
| §9.5, Y9 | **holds, on the user's reading** | (i) 178 calls unconverged, all group (82 at `1e-7`, 96 at `1e-8`): 167 have depth-20 leaves (714, all Clenshaw–Curtis), none below a refused leaf; 11 have no depth-20 leaf. (ii) Of 07b's 198 newly unconverged calls 193 converge and exactly §9.5's five stay unconverged, each untouched and equal to 07b's call. A call with no depth-20 leaf is read as not having "every depth-20 leaf below a refused leaf" (fixed in the census's `predictions.md` before the run; read vacuously, clause (i) would fail on the 11) |

**R32.** `levin-phase-floor-07c-r32-07c-20261119T140417`, 14:04–14:18Z, store primary `4dd39837…`.
`QuadSourceIntegral`'s stage took 11 m 19 s against R31's 11 m 28.6 s. Two group records flipped `phase_limited` False
→ True, each still converged, moved by 5.5×10⁻⁸ of $e_{31} + e_{32}$.

**The census.** `levin-phase-floor-07c-census-20261119T201419`, checkpoint `33d94ce2…`, 18,560 of 18,560 units in
43,080 s, none over budget or in error. It is 07d's "before".
- **Where the stop acts:** in 82,172 of 289,482 calls (28.4 %), 57,659 of them analytic and 24,513 group; 90,176 stops
  and 977,289 stopped leaves. 64.5 % of the stopped leaves are Levin leaves, mostly at depths 3–4; the Clenshaw–Curtis
  ones mostly at depths 11–13; 607,044 were untrusted. 10,800 acted calls still equal 07b's in the five outputs. The
  instrument's counts equal the code's on every call.
- **Convergence against 07b:** 193 group calls newly converged (97 at `1e-7`, 96 at `1e-8`), none newly unconverged,
  no analytic call changed; rows: 136 gained `total_converged`, none lost it. So §C.5's list of newly unconverged calls
  is empty: the 26 LiouvilleGreen draw calls of Part B's Observation 5 have no counterpart in the census.
- **Convergence against 07a:** §9.5's five lost and 07b's five gained; unconverged calls 82 + 96, as at 07a (07b
  179 + 192).
- **abserr against 07b:** a stopped untrusted leaf's declaration can grow a call's abserr a long way: on
  `h-1e-7-1-14470` analytic #7 a depth-3 Levin leaf with $r = 0.99986$ declares $2.46\times10^{-20}$ from
  $\delta_1 = 3.5\times10^{-24}$, 242× 07b's abserr, and the call is still honest; 24 acted analytic calls at `1e-7`
  are above 2× 07b's.
- **Values:** no call moved past the sum of 07c's and 07b's abserrs. Against 07a, 328 group calls moved past the sum
  (07b: 326), item 6's population, left to 07d.
- **Cost:** 0.995 of the central projection and 0.957 of 07b's wall; regions 0.917 of 07b's (`1e-7`: analytic 0.965,
  group 0.883; `1e-8`: analytic 0.976, group 0.891).

**The laptop's re-checks** (2026-11-20), from the copies, against committed records the agents did not write
(`VERIFICATION.md` §7c.5): the copy-back (688 records, 15 run files) by SHA-256; Y4 on R2-28 at both rtols, R4's
`c-1e-7-0-12688-1` and `c-1e-8-3-13490-2` and R5's `m-1e-7-0-1784-g6` against Part A's committed (S) replays and
references; Y6 on `2-14345` #3 and `2-34473` #3 against 06b's committed $S_{10}$ and 07b's committed U6; an untouched
rehearsal row equal in hex in R31, R32 and 07b's committed R31, and the most-moved row's ratios recomputed exactly.
Every figure equals the agents'.

**Deviations in Part C.** The rehearsal agent logged 28: 2 `STRUCTURALLY REQUIRED`, the rest `IMPLEMENTATION CHOICE`,
three of them also `UNINTENDED DRIFT` and harmless (the `acquire` above; two generators' first runs that failed their
own assertions or were caught in the diff before anything ran). The census agent logged 23: 19 `IMPLEMENTATION CHOICE`,
3 `STRUCTURALLY REQUIRED` and 1 `UNINTENDED DRIFT` (failed attempts, each re-run, nothing used from them). The Mac
Studio orchestrator accepted each.

**The user's rulings** (relayed by the laptop orchestrator):
- dispatch: *"Yes, dispatch Part C"*;
- stop 1: *"Yes, continue the rehearsal agent"*;
- stop 2: *"Yes, launch the census"*;
- stop 3: *"Yes, score Y1–Y9 and run the review"*;
- after the review, on Y9's reading and §C.4: *"Yes, accept the Y9 reading and start the copy-back"*;
- on the orchestrator's proposals at §C.4: *"Yes, those rulings stand; make the records commit"*. So:
  1. `[07b-rate-range-rule-chases-noise-leaves-beyond-the-window-stop]` closes: Y9 holds and the honesty predictions
     hold, R2-28 being its own issue;
  2. `[06d-cc-nested-estimate-measures-the-coarse-rule]` is narrowed by §2.3's D2: the fix includes no sharper
     Clenshaw–Curtis estimate, and no candidate estimate bounds CC₄₇'s true error on every sampled leaf; stopped
     Clenshaw–Curtis leaves near $6\pi$ declare CC₂₄'s truncation error (Part B's Observation 4; Part A's item 2);
  3. `[07-c2-gate-threshold-was-never-measured]` is narrowed by §2.3's table by phase variation; the threshold stays
     07d's;
  4. `[07c-chase-stop-acts-on-unreferenced-analytic-calls]` opens, not assigned (the census agent's item 1, the
     review's observation 1);
  5. `[07c-stopped-leaf-declaration-overstates]` opens, not assigned (the census agent's item 2, the review's
     observation 2, Part B's Observation 5);
  6. `[07c-integrand-noise-above-the-cc-floor]` opens, not assigned (Part A's items 1 and 3, Part B's Observation 2).
- The copy-back keeps one of the census's four `runs_hashed_*.txt` listings, whose bodies are equal, and leaves out
  the census's copy of the parent's module, which git holds (`07c/KEPT_ON_MAC_STUDIO.md`).

  R29, R30, R31 and R32 are kept, with 07a's, 07b's and 07c's census run directories and A3 v2's scratch copy.

**Observations not acted on, and not opened as issues** (the agents' `issues.md`, Part A's and Part B's):
- **Part A's items 4–6 and 8–10** (the chase mostly Levin bisection, which the stop reaches; 06d's quoted figure naming
  no case; the descent comparison's "nearest" column; (E)'s seven calls; the references' projection; table 5's
  definition of "touched"): recorded in Part A's log. Items 7, 11 and 12 are
  `[07c-r2-28-trusted-levin-leaf-hides-a-common-error]`'s, opened at Part A's stop.
- **Part B's Observations 1 and 3:** the noise-level chase on the synthetic family with no refusal, where the stop does
  not act by design (the census's unconverged calls are back to 07a's 82 + 96); a refused Levin leaf at the noise level
  is itself stopped, as specified.
- **The rehearsal agent's items:** two rows unconverged at depth 20 in both R31 and R32, from before this prompt (one
  with 5 noise bisections, one short of the limit); captures not byte-identical across trees (unit name, worktree path,
  per-process `str` hashes), so no byte comparison of captures tests an integrand.
- **The census agent's items 3 and 4:** item 6's 328 calls, 07d's (R5 60 of 60 honest at 07c); `compare_07c.py` files
  `route_07a_post_unit` under `call` in its whole-unit categories, a harness record left as written.

## State handed on (Part C)

- The records commit, which this section is part of. No code changed in Part C.
- On the Mac Studio: the R32 store and 07c's census run directory, kept; the worktree `SecondaryGWKit-lpf07c-cc3368c`,
  clean at `cc3368ce`; the registry 96 runs, all finished; the lock free; no Ray process.
- 07d's "before" is 07c's census.

## Part C §5: rehearsal R32 on the Mac Studio

**Prompt:** [`07c-chase-below-a-refused-leaf.md`](../07c-chase-below-a-refused-leaf.md) §5 ("as 07b §5", which is "as 07a
§5"), with §8 and §9 (§9.5's S3 item, §9.7). It ran under Part C's dispatch (`cc3368ce`). The user's word at stop 1:
*"Yes, continue the rehearsal agent"*.

**What was measured:** the code commit `67b5af74`, variant (S).
- **The rule:** below a leaf a rate rule refused, a root stops once its chase has made `_LEVIN_CHASE_STOP_LIMIT = 6`
  noise-level checks within `_LEVIN_CHASE_NOISE_FACTOR = 4` of the yardstick.
- **The parent:** the code commit's parent is `cff4237f`. The parent rehearsal is R31, built at `0a103b5a`, whose code
  is `5e19c4b6`'s.

**Host and trees:** mps026298.
- Every build, replay, score and reader ran in the worktree `SecondaryGWKit-lpf07c-cc3368c`, detached at `cc3368ce`.
  - `levin_quadrature.py` is `d141a80b`.
  - These four are R31's tree's: `QuadSourceIntegral.py` `33cb161e`, the launcher `b0135218`, `main.py` `a5d47f5f`,
    `store_value_diff.py` `7997af85`.
- Nothing was written in the worktree. `status --short --ignored` was empty at every check.
- The main checkout stayed at `28ebaf07`, clean.

**Model:** Claude Opus 5.5, the rehearsal subagent, in three turns:
1. the gate and R32's launch;
2. the end check, the comparisons, and stop 1;
3. S3, the replay, the S3 record, S1, S2, S4, the readers, the release, and these records.

**Date:** 2026-11-19, 14:01–14:52 UTC. **Commit:** none; the laptop commits these records.

**Result:** COMPLETE. Every check of §5 holds.
- **R32 is `done` with `1b332ffa…`, and S0′ holds.**
- **Outside `QuadSourceIntegral` every value table equals R31's** (360,462 rows; 0 unclassified).
- **Inside it, 305 of 864 rows moved.**
  - `total_converged` went up in 1 row and down in none. The one is §9.5's named row.
  - Summed regions went 152,117 → 144,531 (0.950). No row's regions rose.
- **S3 holds.** 0 violations; the largest ratio is 5.25e-6.
- **§9.5's S3 prediction holds.** R31's unconverged record (143 regions) converges in R32:
  - at 86 regions and depth 16;
  - equal bit for bit to Part A's (S) replay record;
  - honest at **0.05331** of its abserr against Part A's `ref-R31-3-20363-g1`.
- **S1, S2 and S4 hold.**
- **The readers** exit 0 with the store unchanged. **The lock is released.**

### What shipped

No code. The records under `var/levin-phase-floor-07c/rehearsal/` (committed by the laptop as
`docs/levin-phase-floor/07c/rehearsal/`): `log.md`, `verification.md`, `issues.md`, `KEPT_ON_MAC_STUDIO.txt`, and:
- **`scripts/`:**
  - adapted from 07b's rehearsal (each with a diff in `out/`):
    - `rec.sh` (`43c59f3b`) and `protected_listing.sh` (`b4d11baa`);
    - `normalise_jobname_labels.py` (`93a0b27e`);
    - `tabulate_qsi_r31_r32.py` (`8e0c91a7`) and `qsi_summary_r31_r32.py` (`907b9950`);
    - `s3_score.py` (`8355d776`);
    - `s1_score_r32.py` (`8609e7fb`), `s2_score_r32.py` (`4a655c7b`) and `block_score_r32.py` (`6debd7ad`);
    - `run_block_reference_r32.sh` (`6db5cd4f`) and `launch_r32.py` (`145c59c8`);
  - unchanged copies:
    - `classify_non_qsi.py` (`26690e61`, 05's);
    - `block_reference_r30.py` (`efac125e`);
    - `run_readers_worktree.sh` (`ec96cf09`);
    - `run_06j.py` (`6227aea4`, 07b Part A's plan driver);
  - new:
    - `s0prime.py` (`17f73ca4`);
    - `s3_record_r32.py` (`4330f394`), §9.5's S3 prediction;
    - `phase_limited_flips.py` (`90a00f22`);
  - **`gen/`**, the generators that made the adapted scripts:
    - `make_tabulate_qsi_r31_r32.py` (`e8be0f00`), `make_qsi_summary_r31_r32.py` (`7190ccf5`) and `make_s3_score.py`
      (`96954584`);
    - `make_scores_07c.py` (`96976b83`) and `make_replay07c_r32.py` (`8cf49456`);
    - `edit_harness_07c.py` (`3a368bbb`);
  - **`capture/`**, 07b's rehearsal's on-disk harness (hashes in `state/capture_harness_r31_sha256.txt`):
    - `prov.py` (`6b65f1a7`) and `worker.py` (`a5fdb18a`), changed where marked "prompt 07c r32";
    - `replay07c_r32.py` (`a4addf1e`), generated from 07b's `replay07b_r31.py`.
- **`out/`:**
  - `commands.log`;
  - the gate, launch, end, stop-1, readers and final checks;
  - `r32_status.json`, `r32-replay_status.json` and `r32_fingerprint_own.txt`;
  - the classifier outputs, the QSI tabulation, its CSVs and its summary, and `phase_limited_flips.*`;
  - `s0prime.*`, `s1_scores.*`–`s3_scores.*`, `s3_record.*` and `s3_record/`, `block_scores.*`, `block_reference/`;
  - the replay's outputs: `cmd/r32-replay/`, `replay_calls.csv`, `replay_calls.checks.json`, `blocks.json`, `refcap.json`;
  - `readers.txt`, `stop1_unconverged_look.txt` and `unconverged_calls_look.txt`.
- **`state/`:** `lock.log`, `deviations_draft.md`, the launch lines, markers and command files, every store-hash file,
  the protected paths, the replay's rows, specs and plan, and `stores/r32/` (symbolic links).
- **Not for copying back:** see `KEPT_ON_MAC_STUDIO.txt`. It lists the two diff JSONs, the protected listings,
  `state/capture/`, `state/inputs/` and `extract/`.
- **The store is kept:** `var/datastores/levin-phase-floor-07c-r32*`.

### Predictions and verdicts

| Rule (prompt 07c §5, as 07b §5) | Scored on | Verdict |
|---|---|---|
| 1. S0′: `git diff --stat 5e19c4b6 cff4237f -- . ':!docs' ':!prompts'` lists nothing the build imports | `CLAUDE.md` only (the `black` paragraph); no `.py`, `.toml`, `requirements.txt` or `config/` | **holds** |
| R32 with R31's command line (slug `r32-07c`, `--prompt 07c`, `--job-name levin-phase-floor-07c`, `--cpus 6`) gives `1b332ffa…` | `done`, exit 0, `1b332ffa…` in `status.json` and again by `runregistrykit store fingerprint` (read only); `classes` equal R31's; about 13.8 min | **holds** |
| 2. The lock: `acquire`, `set-pid` at the launch, `release` after the readers, each a shell command of its own | acquired 14:03:58Z; `set-pid` 19463; released 14:49:20Z after `ray-processes` listed nothing; ten commands, `>>` (deviation 4) | **holds** |
| 3. Outside `QuadSourceIntegral`: every value table identical; every other difference of the four volatile kinds; no row in one store only | 360,462 value-table rows identical; 0 unclassified on the normalised copy (timing, `label` 6,012, `parents_digest` 291, `rho_reuse` 26) | **holds** |
| 3. Inside it: every moved row tabulated, with the rows whose `total_converged` changes in each direction | 305 rows (`out/qsi_rows_r31_r32.csv`); 802 Levin records; `total_converged` 1 up, 0 down | tabulated |
| 4. S3: \|t32 − t31\| ≤ e31 + e32 on every row with finite `total_abserr` in both, or one §9 predicts to move further | 864 scored exactly (Fractions); 0 violations; largest 5.25e-6; the named row 7.0e-8, within the bound | **holds** |
| §9.5's S3 item: R31's record (G+q+r, (1e5, 1e6, 1e6), z 144627.7) converges at 86 regions, honest at 0.053 of its abserr | converged, 86 regions, depth 16; equal to Part A's (S) record; 0.05331484572900594 against `ref-R31-3-20363-g1` (spread 1.1e-6 of abserr, scored); `total_converged` false → true | **holds** |
| 5. S1: V1's twelve locate z ≈ 10211 and are honest against 06g's references | 12 of 12 locate (within 8.52e-6; spacing ≥ 31.0); 12 of 12 honest, largest 0.0670 (w06); 8 equal R31, 4 moved (one chase stop each) | **holds** |
| 5. S2: P1's two rows locate within one spacing and are honest against `p1_references` | \|x − x_s\| 7.1e-9 and 5.6e-9 against 4.12e-3; ratios 0.0202 (= R31) and 0.111 (R31 0.111) | **holds** |
| 5. S4: R29's four blocks each locate their root, guard no leaf, and are honest against R29's block references | 4 of 4: root bit-equal, 0 guarded, 4.37e-5 to 6.17e-4; each equal to R31 and R28 bit for bit | **holds** |
| 6. The readers on R32 exit 0, store byte-identical; lock released, `ray-processes` empty; R31 and R32 kept | 6 of 6; identical; free; nothing listed; R31 equal to its KEPT hashes | **holds** |

**§9.5's S3 item, in detail** (`out/s3_record.txt`, `out/s3_record.json`):

| | R31 (store) | R32 (store) | R32's call (replay at the tree) | Part A's (S) record |
|---|---|---|---|---|
| value | `0x1.44ad9d596748dp-62` | `0x1.44ad9d59665e8p-62` | `0x1.263e9f583ee54p-79` | `0x1.263e9f583ee54p-79` |
| abserr | `0x1.74349fae30cb8p-87` | `0x1.741c57c6295e9p-87` | `0x1.513b2851b5963p-104` | `0x1.513b2851b5963p-104` |
| converged | False | True | True | True |
| regions / max depth | 143 / — | 86 / — | 86 / 16 | 86 / 16 |
| refusals / chase | — | — | range 4, edge 0 / 2 stops, 28 stopped leaves, 12 noise bisections | — |

- Store values are (1 + z) times the call's.
- **The integrand is the reference's.** `block_reference_r30.py`, run on R32's capture and on R31's capture (07b's
  rehearsal, `df2d478f…`, the one Part A read), gives:
  - 1,001 samples bit-identical, with x_span and kw equal;
  - R31's capture, replayed at the tree, reproduces R32's call.
- **The reference** is Part A's split FCC r2, 1.9015112786324759e-24. Its spread over four readings is 7.24e-38, 1.1e-6 of
  the abserr, so the call is scored.
- **The verdict:** \|v − ref\| / abserr = 0.05331484572900594, the same as Part A's (S) ratio. **Honest.**

**The inside-QSI tabulation** (`out/qsi_tabulation_r31_r32.txt`, `out/qsi_summary_r31_r32.txt`):
- **Rows:** 559 unmoved, 305 moved. `total` moved in 251, `total_abserr` in 252, `analytic_rad` in 106.
- **Largest changes:**
  - \|Δtotal\| is at most 1.050e-5 of R31's abserr and of R32's, at (1e6, 1e5, 1e6), z 437173.17. The median over the
    251 is 1.7e-7.
  - \|Δanalytic_rad\| is at most 6.48e-4 of `analytic.abserr`.
  - **Group records:** 477 values moved; \|Δv\| is at most 1.67e-3 of either abserr; none is beyond e31 + e32.
- **Regions:**
  - Over the moved rows, regions fell in 233, rose in 0, and were equal in 72. The per-row ratio where changed runs
    0.634–0.998, median 0.939.
  - Maximum depth changed in 78 rows. The rows at depth 20 went from 7 to 5.
  - Group-record regions over the differing records went 45,439 → 37,853, and analytic-block regions 3,963 → 3,367.
  - QuadSourceIntegral's stage took 11 m 19 s against R31's 11 m 28.6 s.
- **Convergence:**
  - `total_converged` went 861 → 862: (1e5, 1e6, 1e6), z 144627.7, the named row.
  - Unconverged group calls went 3 → 2. The two rows still unconverged were unconverged at R31 too (`issues.md` item 1).
  - No analytic block's `converged` changed.
- **Location:** unchanged. Group records: 44 located points, 562 window pieces, 0 guarded regions. Analytic blocks:
  400 / 2,515 / 0. 202 rows hold a located point.
- **`phase_limited`:** two group records flipped False → True, and none flipped back. They are tabulated below, at the
  orchestrator's note; descriptive (`out/phase_limited_flips.txt`).

  | (k, q, r; z) | record | label | phase_limited | converged | regions | \|Δv\|/e31 | \|Δv\|/e32 | \|Δv\|/(e31+e32) |
  |---|---|---|---|---|---|---|---|---|
  | (1e5, 3.162e5, 3.162e5; 755.687) | `subintervals[1].groups[1]` | G+q−r | False → True | True → True | 38 → 36 | 1.09e-7 | 1.09e-7 | 5.47e-8 |
  | (1e5, 3.162e5, 3.162e5; 755.687) | `subintervals[1].groups[2]` | G−q+r | False → True | True → True | 38 → 36 | 1.09e-7 | 1.09e-7 | 5.47e-8 |

  Both records have the same value, `0x1.5535a1169116bp-67` → `0x1.5535a116911a6p-67`, and the same abserr, because
  q = r makes the two integrands one. abserr32/abserr31 is 1.000000. Neither has a located point.

**The replay** (`out/cmd/r32-replay/`; run `levin-phase-floor-07c-r32-replay-20261119T143558`, 8.5 min):
- **Reproduction:**
  - 445 of 445 records reproduce R32 bit for bit, with Ray uninitialised in all 203 workers.
  - The four blocks reproduce in passes 1 and 2.
  - The capture of the record of interest reproduces R32.
- **Over the 4,180 replayed calls (932 group, 3,248 analytic):**
  - Range-rule refusals: 567 in 359 group calls, and 6,517 in 2,587 analytic calls. The edge rule refuses nothing.
  - **Chase stops:** 180 in 164 group calls, with 4,061 stopped leaves and 1,589 noise bisections. 225 in 225 analytic
    calls, with 1,539 stopped leaves and 8,888 noise bisections.
  - **No call has a chase stop without a rate-rule refusal.** Checked call by call over all 4,180 calls; 0 found
    (`out/chase_without_refusal_look.txt`).

### Deviations from the prompt

1. **`rec.sh` is 07b's** (`9a80a313`) with its record path changed by sed. IMPLEMENTATION CHOICE.
2. **S0′ is scored by a new `s0prime.py`.**
   - "Nothing the build imports" is read as: no listed `*.py`, `*.toml`, `requirements.txt` or `config/` path.
   - The raw `git diff --stat` is also in `out/00-gate.txt`.
   - IMPLEMENTATION CHOICE.
3. **The lock commands ran from the worktree with `PYTHONPATH` set**, each with an `echo` header (a builtin) and output
   `>>` into `state/lock.log`, as 07b's did. IMPLEMENTATION CHOICE.
4. **The `acquire` ran in the same zsh command as an earlier heredoc `cat >> out/commands.log`.**
   - The `cat` had exited before the `acquire` started.
   - Strictly, the acquire was not a lock command in a shell of its own.
   - It acquired normally. UNINTENDED DRIFT, harmless.
5. **A listing of every protected file** (mtime, size, path) was taken before, at stop 1 and at the end.
   - Its paths include `var/levin-phase-floor-07c/review/`, the orchestrator's directory. Its 13 changed lines were
     expected.
   - IMPLEMENTATION CHOICE.
6. **The R32 launch line is 07b's form, run in zsh.** The purpose was held in a shell variable and passed as `sh`'s `$0`;
   its text is also in `state/purpose_r32.txt`. IMPLEMENTATION CHOICE.
7. **The fingerprint was also re-derived** by `runregistrykit store fingerprint` without `--write`. Read only; the store
   re-hashed equal. IMPLEMENTATION CHOICE.
8. **`normalise_jobname_labels.py` is 07b's**, with its prefixes (07a- → 07b-, 07b- → 07c-) and docstring changed.
   IMPLEMENTATION CHOICE.
9. **The classifier was run on the raw and the normalised JSON.**
   - The raw gives 5,148 UNCLASSIFIED, every one a `label` whose job-name prefix differs.
   - The verdict is the normalised copy's.
   - STRUCTURALLY REQUIRED.
10. **The classifier prints no provenance header,** so the shell wrote one. IMPLEMENTATION CHOICE.
11. **Stdlib-only scripts ran with `-B` and `PYTHONPATH`, cwd the worktree, not with `-I`.** The generators ran with
    `-I`. None imports tree code. IMPLEMENTATION CHOICE.
12. **`tabulate_qsi_r31_r32.py` is generated from 07b's** by `gen/make_tabulate_qsi_r31_r32.py`.
    - It changes 72 store tags, TREES, RUNS, the DEPTH_MAX comment and the docstring. The verbatim reader is asserted
      unchanged. IMPLEMENTATION CHOICE.
    - The generator's first run aborted on its own assertion (a wrong line index) before writing anything. It was
      corrected by two asserted replacements and re-run. UNINTENDED DRIFT, harmless.
13. **`qsi_summary_r31_r32.py` is generated from 07b's** (161 tag shifts, the tree path, the docstring).
    - `additions_07b` is kept under its name, and nothing new was added.
    - Its "record-level S3 form" uses float arithmetic and is descriptive only; S3 is the exact `s3_score.py`.
    - IMPLEMENTATION CHOICE.
14. **A read-only look at the unconverged rows was saved** (`out/stop1_unconverged_look.txt`), not printed and lost.
    IMPLEMENTATION CHOICE.
15. **The stage timings of R31 and R32 were diffed** into `out/stage_timings_r31_r32.diff`. IMPLEMENTATION CHOICE.
16. **S3 is scored by 07b's `s3_score.py`,** generated with the tags shifted.
    - §9.5's named row is listed whether or not it is beyond the bound, and is not counted as a violation. It is within
      the bound anyway.
    - IMPLEMENTATION CHOICE.
17. **The capture harness is 07b's on-disk copy.**
    - `prov.py` points at the tree and R32, and `worker.py` adds `chase_07c`.
    - `replay07c_r32.py` is generated from 07b's driver (79 tag shifts).
    - Its records of interest follow 07b's rule. There is 1 record: the converged change.
    - IMPLEMENTATION CHOICE.
18. **`make_s3_score.py`'s first output shifted the reader module's name twice** (`tabulate_qsi_r32_r32`). It was caught
    in the diff before any run; the generator was corrected and re-run. UNINTENDED DRIFT, harmless.
19. **The replay ran as a registered run** (slug `r32-replay`), by 07b's `run_06j.py` and a new `launch_r32.py`, as 07b's
    `r31-replay` did. `run_06j.py` appends each command's output to `out/commands.log`. IMPLEMENTATION CHOICE.
20. **Block pass 1 (the pilot) ran in session**, 56.6 s, as 07b's did. IMPLEMENTATION CHOICE.
21. **§9.5's S3 record is scored against 07c Part A's `ref-R31-3-20363-g1`.**
    - These are Part A's readings of 07b's R31 capture of that call.
    - It is not a reference in 07b's rehearsal: 07b captured the call but read no reference for it.
    - No new reference was made.
    - IMPLEMENTATION CHOICE, accepted by the orchestrator.
22. **The two `phase_limited` flips are tabulated by a new `phase_limited_flips.py`**, at the orchestrator's note.
    IMPLEMENTATION CHOICE.
23. **S1, S2 and S4 are 07b's scripts, generated with the tags shifted** (`gen/make_scores_07c.py`).
    - The parent's numbers are read from 07b's score files.
    - R32's block serials are used (0-10172, 1-20068).
    - Each call's `chase_07c` is recorded.
    - S4's integrand is still compared with R29's capture, on which the references were taken.
    - IMPLEMENTATION CHOICE.
24. **The S3 record's integrand is compared by samples, not by blob.**
    - The plan was byte equality of R32's and R31's captures.
    - The blobs differ in the unit name (it includes the store serial), the worktree's path and the per-process `str`
      hashes. 07a's and 07b's block captures differ from their parents' in the same way (`issues.md` item 2).
    - So `block_reference_r30.py`, unchanged and without `--reference`, was run on both captures. Its 1,001 samples,
      x_span and kw were compared. The byte comparison is still reported, as False.
    - STRUCTURALLY REQUIRED.
25. **`block_reference_r30.py` replayed 07b's R31 capture at this tree.** It read the capture only, and wrote its output
    under `out/s3_record/`. IMPLEMENTATION CHOICE.
26. **The S3 record's ratio is evaluated in exact Fractions** of the double and the decimal readings, where Part A used
    mpmath at 40 digits. Both give 0.05331484572900594. IMPLEMENTATION CHOICE.
27. **The readers ran in session** (136.6 s), not registered, as 07b's did. IMPLEMENTATION CHOICE.
28. **Two read-only looks at the replayed calls were saved:** the unconverged calls (`out/unconverged_calls_look.txt`),
    and the calls with a chase stop but no rate-rule refusal (`out/chase_without_refusal_look.txt`; none).
    IMPLEMENTATION CHOICE.

### Verification performed

- **Gate** (`out/00-gate.txt`, `out/s0prime.*`, `out/01-registry-list.txt`, `out/02-stores-before.txt`, `state/lock.log`):
  - the worktree at `cc3368ce`, clean with `--ignored`; the main checkout at `28ebaf07`, clean;
  - `67b5af74^` = `cff4237f`;
  - `67b5af74..cc3368ce` outside docs and prompts is empty;
  - `0a103b5a..cc3368ce` is empty under every path the build imports outside `AdaptiveLevin/`;
  - 92 runs, all finished; the lock free; no Ray process; `auth_token` mode 600;
  - R29, R30 and R31 equal to 07b's final hashes, and R31 equal to its KEPT list.
- **R32** (`out/r32_launch_check.txt`, `out/r32_end_check.txt`, `out/r32_fingerprint_own.txt`):
  - launched 14:04:16Z (sh 19451, driver 19463);
  - `done` by 14:18:18Z; no Traceback, WARNING or ERROR.
- **Stores:**
  - R29, R30, R31, R31's run directory and A3 v2 byte-identical throughout (`state/stores_sha256_final.txt`);
  - R32 unchanged after its build, across the fingerprint, the comparisons, the replay, the scoring and the readers.
- **The replay:** 445 of 445 records reproduce R32 bit for bit; the blocks and the record capture reproduce.
- **The lock:** ten commands, each exit 0, ending free with nothing listed.
- **Final** (`out/final_check.txt`):
  - 94 runs, all finished; `~/.raypoolkit` empty; no Ray process;
  - both trees clean and unmoved, nothing newer in the worktree;
  - the protected listing differs only under the orchestrator's `var/levin-phase-floor-07c/review/`;
  - no `__pycache__` in the records.

### Observations not acted on

1. **Two rows stay unconverged at depth 20 in both stores** (`issues.md` item 1).
   - One of them, (3.162e5)³ at z 573.11, G−q−r, was replayed: 2 range refusals, 0 chase stops, 5 noise bisections.
     That reads as one check short of the stop.
   - Its bearing is on §6's Y9. The census's instrument decides.
2. **Capture blobs are not byte-reproducible across trees** (`issues.md` item 2).
3. **The two `phase_limited` flips** are tabulated above. Both records stay converged, and they moved by 5.5e-8 of
   e31 + e32.
4. **The readers subsample k and z at random**, so their file counts differ from R31's (3,122 against 3,189), as 07a
   and 07b recorded. They print the same six shard-count warnings and two `MatplotlibDeprecationWarning`s as 07b's.
5. **`out/commands.log` holds the plan driver's appended command outputs** (deviation 19).

### State handed on

- **R32:** `levin-phase-floor-07c-r32-07c-20261119T140417`, `done`, `1b332ffa…`. Its store is kept.
- **The replay run:** `levin-phase-floor-07c-r32-replay-20261119T143558`, `done`, exit 0.
- **R31, R30 and R29** are unchanged and kept.
- **The lock is released.** Nothing of this agent runs. Both trees are clean and unmoved.
- §6, the census, was not touched.

## Part C §6: the census at 07c's code (Mac Studio), the census agent

**Subject:** prompt 07c §6, as amended by §9 (§9.4, §9.5, §9.7). The census was re-run at the code commit `67b5af74`
(variant (S), the stop below a refused leaf), with 07b's census as the "before". It ran on the Mac Studio (`mps026298`),
2026-11-19 to 2026-11-20, by Part C's census agent. This log covers Gate Y0, the predictions, the projection, the census,
stop 3's two checks (Y1's untouched calls and Y8), Y1–Y9, the descriptive items and the records. Ray-free, no lock, no
store built. The sections below are in the order the work was done; each stop's section is kept as written then.
- **The tree h:** the worktree `SecondaryGWKit-lpf07c-cc3368c`, detached at Part C's dispatch SHA `cc3368ce`.
  - Its code is `67b5af74`'s: `AdaptiveLevin/levin_quadrature.py` `d141a80b…`.
  - It stayed clean with `--ignored`, with no `__pycache__` (`out/cmd/01`, `29`). The main checkout stayed at `28ebaf07`,
    clean.
- **The parent:** there is no parent worktree. Y0's K0 run at the parent ran in h with the parent's module installed in
  place of h's by `variant_loader.py`: `state/variants/levin_quadrature_parent.py`, from `git show cff4237f`, `8956c7b6…`,
  blob `adb7b7df`, the same as 07b's code commit `5e19c4b6` and 07b's census tree `0a103b5a` (`out/cmd/03`).
- **The "before":** 07b's census, `levin-phase-floor-07b-census-20261117T010200`, checkpoint `396784bc…`.
- **The store:** 06's certified scratch copy of A3 v2, read only.

**Commit:** none here. The orchestrator commits these records from the laptop as `docs/levin-phase-floor/07c/census/`.

**The user's rulings, relayed by the orchestrator:**
- At stop 2: *"Yes, launch the census"*.
- At stop 3: *"Yes, score Y1–Y9 and run the review"*.

**Result:**
- **Gate Y0 holds**, all five clauses.
- **The projection** was 43,312 s central and 55,392 s at its widest margin times 07b's shortfall, within §9.4's 48 hours.
- **The census** is `levin-phase-floor-07c-census-20261119T201419`, `done`:
  - 18,560 of 18,560 units, 0 in error, 0 over budget;
  - wall 43,080 s (11.97 h; 07b: 45,016 s; 0.995 of the central projection).
- **Stop 3's two checks pass:**
  - all 207,310 untouched calls equal 07b's in Y1's five outputs;
  - all 453 Y8 ids equal (S)'s replay records.
- **Y1–Y9 all hold.** Y4 holds with R2-28 converged and dishonest at 1.1434 of its abserr, as §9.5 predicted, reported
  against `[07c-r2-28-trusted-levin-leaf-hides-a-common-error]`.
- **No call is newly unconverged** against 07b's census. 193 group calls are newly converged.
- **The store, 07b's census and K0 runs, 07b's records and Part A's records are unchanged**, before, at each stop and at
  the end.

### What shipped (at stop 2; the final additions are in "Y1–Y9 scored" below)

No code. The records are in `var/levin-phase-floor-07c/census/`.
- **`scripts/`:** 07b's census harness, copied from h's `docs/levin-phase-floor/07b/census/scripts/`, byte-identical to
  07b's Mac Studio copy (`out/cmd/00`: 232 files, list hash `c74a7107…` for all three). Changes are marked "prompt 07c".
  The diffs are `out/scripts_diff_before_k0.txt` and `out/scripts_diff_at_stop2.txt`.
  - **Changed:**
    - `prov.py`: the tree h (`cc3368ce`), the parent's module, 07b's census and K0 runs;
    - `worker.py`: the three counts on every call; the 07c tree evidence; the stop's instrument (`chase_07c`) and its
      switch;
    - `run_k0.py`: the parent variant 07c's; `--no-instrument` also turns off the stop's instrument;
    - `run_census.py`: its docstring only (07b's 48 h backstop kept).
  - **New:** `ast_check_07c.py`, `y0a_imports_07c.py`, `index_07b_07c.py`, `compare_07c.py`, `select_y0_units_07c.py`,
    `launch_07c.py`, `census_projection_07c.py`, `write_predictions_07c.py`, `hash_runs_07c.py`, and, not yet run on the
    census, `launch_census_07c.py`, `score_y8_07c.py`, `kept_list_07c.py`.
- **`out/`:** every command and its output (`out/cmd/`, `out/commands.log`):
  - Y0: `out/y0/y0a_ast.txt`, `out/y0/y0a_imports.txt`, `out/k0/compare_y0b.json`, `out/y0c3/compare_instr.json`,
    `out/y0c4/compare_det.json`, and `out/k0/compare_k0_h_calls.json` (descriptive);
  - the projection: `out/census_projection.json`;
  - before and at stop 2: `out/runs_hashed_{before,stop2}.txt`, `out/cmd/04`–`06`, `27`–`28b`.
- **One registered run, `done`:** `levin-phase-floor-07c-k0-20261119T151009`, 11 of 11 commands, exit 0, checkpoint
  `1cdc934e…`.

### The stop's instrument (worker.py, "prompt 07c")

§9.5 asks, for each stopped leaf, its route, depth, declaration, `r` and its share of its call's abserr. The stopped leaf
is made inside `_adaptive_levin`'s closures (`_stop_chase_leaf`), which no module attribute reaches. So:
- a pass-through at the module attribute `_chase_at_noise_level` (looked up in the module's globals at call time) returns
  exactly what the original returns;
- on its first call within a recorded Levin call, it finds that call's `_adaptive_levin` frame and replaces, in the
  frame's cell, `_stop_chase_leaf` by a pass-through that calls the original with the same leaf;
- after the original returns, it records the leaf if the original stopped it (its region not floor-limited before, and
  floor-limited after: the original's own test): seq, span, depth, the region's type (Levin or `direct`, the
  Clenshaw–Curtis pair), `forced_direct`, `window_index`, `guarded`, the trust before the stop, `r`, δ₁, δ₂, the check's
  ν_B, the residual, the declared resolution and e′ after the stop, and the root;
- after the call it reads, never writes, the frame's `chase_stopped`, `chase_counts` and `chase_root`. The share is the
  stopped region's e′ as returned (`budget_err`) over the call's abserr. Per call it records the leaves by route and depth,
  the stopped leaf carrying the most of the abserr, and whether the instrument's counts equal the three result counts
  (stopped leaves, roots stopped, bisections counted).

Shown to change nothing by Y0 (3). Its counts equal the result's on every call of Y0 (3)'s two units.

### Gate Y0 and the predictions (at stop 2)

`state/predictions.md` holds four parts of the prompt, each copied verbatim **by text** and checked as a substring:
- §6's Y1–Y8 and descriptive items (prompt lines 502–521);
- §9.5's additions, Y8 and Y9 among them (663–697);
- §9.7's stops (720–724);
- §8's Part C stops (538–539).

Part 5 is this agent's reading of each rule, fixed before the run. Part 6 states Y0 and the projection as they stood.
Y1–Y9 are not yet scored.

| Prediction | Verdict | The numbers it was scored on |
|---|---|---|
| **Y0 (1)** the parent's code is 07b's (S0′) | **holds** | `git diff --stat 5e19c4b6 cff4237f -- . ':!docs' ':!prompts'` lists `CLAUDE.md` alone. `levin_quadrature.py` at `cff4237f` is byte-identical to `5e19c4b6`'s (`8956c7b6…`). The K0 workers at the parent loaded 101 files of h, none differing between `cff4237f` and h or between 07b's census tree `0a103b5a` and h; h's own `levin_quadrature.py` was not loaded; the variant was the parent's `8956c7b6` (`out/y0/`) |
| **Y0 (2)** the harness reproduces 07b's census at the parent | **holds** | 100 of 100 K0 units are bit-identical to 07b's census: 2,110 calls (566 group), 55,032 returned regions. Only 07c's five new keys were set aside, 0 of them in 07b's records; 15 workers' evidence all holds (`out/k0/compare_y0b.json`) |
| **Y0 (3)** the instrument changes nothing | **holds** | `h-1e-8-0-33929` (the fix acts: calls #2 and #3 each 1 stop, 10 stopped leaves, 6 bisections counted) and `h-1e-8-3-14505` (21 calls, every `num_chase_stops` 0, 20 with a counted bisection; chosen by the rule fixed before the run) are IDENTICAL with the instruments on and off. Every call of the run ON carries the stop's instrument with its counts equal to the result's; no call of the run OFF does (`out/y0c3/compare_instr.json`) |
| **Y0 (4)** the new code is deterministic | **holds** | `h-1e-8-0-33929`, run twice in two fresh workers, is identical on every key (`out/y0c4/compare_det.json`) |
| **Y0 (5)** §9.4 the projection within 48 h | **holds** | Central 43,312 s (schedule model, ρ by rtol and the 720; ρ_h 0.957–0.994 by class, 0.965 overall). Highest margin 46,285 s. Times 07b's shortfall 1.197: 51,833 s central, 55,392 s highest. No unit projected over budget (`out/census_projection.json`) |

**Descriptive, from K0 at h** (`out/k0/compare_k0_h_calls.json`; not a prediction): the fix acts in 591 of 2,110 K0
calls: 163 of 566 group calls and 428 of 1,544 analytic calls. All 1,519 untouched calls equal 07b's in Y1's five outputs
and in the whole record; the 19 units untouched in every call equal 07b's in both totals. 79 acted calls equal 07b's in
the five outputs. No call changes `converged`. Regions 51,687 against 07b's 55,032 (0.939).

### Deviations from the prompt (so far)

1. **The harness changes** (`prov.py`, `worker.py`, `run_k0.py`, `run_census.py`'s docstring;
   `out/scripts_diff_at_stop2.txt`). STRUCTURALLY REQUIRED for the tree, the parent and the three counts.
2. **The parent ran as a variant from `state/variants/` in h**, as the dispatch directs. STRUCTURALLY REQUIRED.
3. **New scripts** (listed under What shipped). IMPLEMENTATION CHOICE.
4. **The stop's instrument replaces a closure in the running call's frame** (described above). No module attribute
   reaches `_stop_chase_leaf`, and the alternative (reconstructing which leaves were stopped from the returned regions)
   cannot tell a leaf the stop floor-limited from one already floor-limited. Y0 (3) shows it changes nothing.
   IMPLEMENTATION CHOICE.
5. **Y0 (3)/(4)'s units:** `h-1e-8-0-33929` was fixed by name (two of Part A's chase calls, `c-1e-8-0-33929-2` and
   `-3`, which (S)'s replays touched; the cheapest such unit at 07b, 6.3 s). The quiet unit was chosen from K0 at h by a
   rule fixed before the run: every call untouched; then the most calls with a counted bisection (so the instrument's
   pass-through ran without a stop), the most calls, the most group calls. IMPLEMENTATION CHOICE.
6. **`--no-instrument` turns off 07a's, 07b's and 07c's instruments together.** Mode `instr` sets aside the three
   instruments' keys but not the counts. IMPLEMENTATION CHOICE.
7. **The projection** is 07b's model, re-based on 07b's census, with 07b's shortfall (measured over central, 1.197)
   applied as the dispatch asks. IMPLEMENTATION CHOICE.
8. **`predictions.md` adds §9.7's and §8's Part C blocks, and parts 5–6.** IMPLEMENTATION CHOICE.
9. **Smoke tests ran in the session** (`state/smoke/`, `out/cmd/11`–`17b`), not scored. They include a dry run of the Y8
   scorer against 07b's census (`17b`: 128 of 453 equal, as expected of a code that is not (S); it exits 1 by design).
   IMPLEMENTATION CHOICE.
10. **Part A's whole `measurement/` was hashed** (17,111 files), with 07b's K0 run directory and 07b's census working
    records, in place of "Part A's `measurement/`" alone. IMPLEMENTATION CHOICE.
11. **The registry has no job-name field.** The run ids carry the prefix `levin-phase-floor-07c`. STRUCTURALLY REQUIRED.
12. **The A3 v2 fingerprint was not recomputed.** The copy is checked by SHA-256 against its certification, as 07b did.
    IMPLEMENTATION CHOICE.
13. **Failed attempts, each re-run.** UNINTENDED DRIFT (failed attempts). The first two smoke comparisons were written
    with `$T:scripts` and never ran: zsh read `:s` as a modifier ("bad substitution"). They left no record, and were
    re-run with the path quoted as `out/cmd/14` and `15`.
14. **The readings of Y4, Y8 and Y9 in `predictions.md` part 5** (fixed before the run). IMPLEMENTATION CHOICE:
    - Y4's "scored" test takes min(abserr) over this census and 07b's (07b's U4 form, one code on), with 07a's as well for
      R5, as Part A's R5 rule took it;
    - Y8's population is (S)'s 454 replay records less R31's record (not a census call): 453 ids, 448 census calls;
    - Y9 (i) does not count a call with no depth-20 leaf as "every depth-20 leaf below a refused leaf".
15. **Smoke runs 12 and 13 ran concurrently** (two workers in all). IMPLEMENTATION CHOICE.

### Verification performed (so far)

`verification.md` has the account. In brief:
- **The trees:** h is at `cc3368ce`, clean with `--ignored`, no `__pycache__`. The main checkout is at `28ebaf07`, clean.
- **The scratch copy:** EQUAL to its certification and to its before-hash at stop 2. `shasum -c` against 06e's after-hashes
  is OK. No `-wal` or `-shm` files.
- **07b's census and K0 runs, 07b's records and Part A's records:** 150 of 150 and 5 of 5 files equal to 07b's KEPT list;
  07b's 54 kept working files equal; all 17,111 Part A files and 517 07b working files byte-identical before and at
  stop 2.
- **The registry:** 95 runs, all finished. No lock was taken.

### Observations not acted on (so far)

1. **The stop acts on analytic calls in K0:** 428 of 1,544 analytic K0 calls, more than on group calls (163 of 566).
   §9.3 says Part A's sample had no chase on an analytic call, and scores the stop there only by R2's 21 and R3's
   analytic calls. None of the 428 changes `converged` in K0.
2. **The machine has 10 cores** and ran with a background load of about 3–4 cores (Spotlight, CategoriesService) during
   K0. The parent's K0 units ran 1.018× 07b's census unit time, so the projection carries about 2% for it.

### State handed on (at stop 2)

- **Nothing is running.** The k0 run is `done`. No lock was taken.
- **The census is not launched.** It waits on the user's word: `scripts/launch_census_07c.py 43312 46285`.
- **Read only and unchanged:** the scratch copy, 07b's census and K0 runs and records, Part A's records, by hash.

### The launch (after stop 2)

- **The user's word at stop 2**, relayed by the orchestrator: *"Yes, launch the census"*.
- `out/cmd/31`: the registry before the launch, 95 runs, all finished.
- `out/cmd/32`: `scripts/launch_census_07c.py 43312 46285` at 2026-11-19T20:14:19Z.
  - Run `levin-phase-floor-07c-census-20261119T201419`; `sh` pid 90730; driver pid 90737.
  - `run_census.py` `a4ac7ba7…`, `worker.py` `b1fd6cf8…`; module `d141a80b…`, no variant.
- `out/cmd/33`: one check, 36 s after the launch.
  - `state` running; pid 90737 answers `kill -0`; heartbeat 2026-11-19T20:14:20Z.
  - Six 1e-7 workers started; stderr empty.
  - No checkpoint line yet.

### Planned stop 3 (the census `done`; the two checks of §C.1, and the descriptive list)

**The run.** `levin-phase-floor-07c-census-20261119T201419` is `done`, exit 0 (`out/cmd/40`):
- 18,560 of 18,560 units, 0 in error, 0 over budget;
- `stderr.log` empty; checkpoint 3,498,056,160 bytes;
- wall 43,080 s (11.97 h), from the manifest's `created` (2026-11-19T20:14:19Z) to the last heartbeat
  (2026-11-20T08:12:19Z).

**Check 1: Y1's untouched calls** (`out/stop3/compare_census_calls.json`, `.calls.jsonl`; `out/cmd/46`). Every one equals
07b's call in all five outputs. No stop.

| rtol, kind | untouched (`num_chase_stops` 0) | of these equal to 07b's | touched | touched but equal in the five |
|---|---|---|---|---|
| 1e-7 analytic | 73,021 | 73,021 | 34,395 | 5,707 |
| 1e-7 group | 24,775 | 24,775 | 12,550 | 206 |
| 1e-8 analytic | 84,152 | 84,152 | 23,264 | 4,684 |
| 1e-8 group | 25,362 | 25,362 | 11,963 | 203 |
| **all** | **207,310** | **207,310** | **82,172** | **10,800** |

- Every untouched call's whole record also equals 07b's (category `none` on all 207,310).
- 8,508 units are untouched in every Levin call. All 8,508 equal 07b's in `total` and `total_abserr`.
  - 8,495 of them equal 07b's whole.
  - 13 differ only in the harness's own post-section re-solves: `analytic_06h.parent_counterfactual` (11) and
    `route_07a_post_unit.n_solves` (11). These run Levin at the new code after the timed section, and are not among Y1's
    outputs.
- Every call carries the three counts (0 `no_counts`).
- All 18,560 units: module `d141a80b` from the tree, no variant, h clean at its SHA, no foreign module. All 16 workers'
  evidence holds.

**Check 2: Y8** (`out/stop3/y8.json`; `out/cmd/47`).
- 453 of 453 ids, naming 448 distinct census calls, equal (S)'s replay records in value, abserr (hex), `converged`,
  `num_regions` and max depth.
- Each record equals its `S_summary_vplain.json` row. All were replayed under module `09088683…`.
- The stop acted in 327 of the 453. No stop.

**The stop's instrument** (`out/cmd/48b`): on all 289,482 calls its counts equal the three result counts. None is in
error, none failed to find its frame, and every stopped leaf is among the returned regions.

**Descriptive: `converged` against 07b's census** (`out/stop3/stop3.json`, `out/stop3/newly_unconverged.md`;
`out/cmd/48`). Nothing is scored.
- **Newly unconverged (converged at 07b, unconverged here): none,** at either rtol, analytic or group. The list is empty.
- **Newly converged:** 193, all group calls: 97 at 1e-7 and 96 at 1e-8.
- **Unconverged under both codes:** 82 group calls at 1e-7 and 96 at 1e-8.
- **No analytic call is unconverged** under either code (107,416 at each rtol).

**Cost** (descriptive):

| | 07c | | | 07b's census |
|---|---|---|---|---|
| census wall | 43,080 s | 0.995 of the central projection (43,312 s); 0.831 of central × 1.197 (51,833 s); 0.931 of the highest margin (46,285 s) | | 45,016 s (07c / 07b 0.957) |
| unit wall, 1e-7 | 49,405 s | | | 52,154 s |
| unit wall, 1e-8 | 49,325 s | | | 54,667 s |
| largest unit wall | 40.0 s | | | 40.1 s |
| regions, all calls | 5,536,209 | | | 6,039,639 (0.917) |
| regions by rtol and kind | 1e-7 analytic 0.965, 1e-7 group 0.883, 1e-8 analytic 0.976, 1e-8 group 0.891 | | | |

**Read only and unchanged after the census** (`out/cmd/41`–`45`):
- The scratch copy is EQUAL to its certification and to its before-hash; `shasum -c` gives 5 OK.
- 07b's census and K0 runs, 07b's records and Part A's records all hold, and are EQUAL before and after.
- h is clean with `--ignored`, with no `__pycache__`; the main checkout is at `28ebaf07`, clean (`out/cmd/49`).
- The registry holds 96 runs, all finished.

**Deviations at stop 3** (numbered on from 15):

16. **`stop3_07c.py`** is new, after 07b's `stop3_cost_07b.py`: Y1's counts, `converged` against 07b, the newly
    unconverged list with each call's carrying stopped leaf, and cost. IMPLEMENTATION CHOICE.
17. **The instrument's count check** (`out/cmd/48b`) was first run outside `rec.sh` into `out/stop3/`. That file was
    deleted and the check re-run through `rec.sh`. IMPLEMENTATION CHOICE.
18. **Stop 3's checks ran in the session through `rec.sh`**, not as registered runs. The longest took 2.4 minutes.
    IMPLEMENTATION CHOICE.

**Observation not acted on:**

3. **The 13 untouched-in-every-call units that differ from 07b's whole record do so only in the harness's own
   post-section re-solves** (`analytic_06h.parent_counterfactual`, `route_07a_post_unit.n_solves`). These run at the
   new code and are not census outputs. compare_07c.py files `route_07a_post_unit` under category `call` because 07b's
   regex does not name it; that is a categorisation gap, not a change in a census call.

### Y1–Y9 scored (after stop 3, on the user's word)

State: `state/predictions.md` (`cad4fd7f…`, unchanged since 2026-11-19T15:30:24Z). Each verdict is against its rule as
written there (part 5's reading) and in the prompt. Scorers are listed under deviation 19. The summary is
`out/scores/summary.json` (`out/cmd/55`).

| Prediction | Verdict | The numbers it was scored on |
|---|---|---|
| **Y1** nothing else moves | **holds** | 207,310 untouched calls (`num_chase_stops` 0): 97,796 at 1e-7 (73,021 analytic, 24,775 group) and 109,514 at 1e-8 (84,152, 25,362). All equal 07b's in value, abserr, `converged`, `num_regions` and max depth (hex), and in the whole call record. The 4,233 + 4,275 = 8,508 units untouched in every call equal 07b's in `total` and `total_abserr`. 0 calls lack the counts. Coverage 71.6% of calls. Every one of 07b's 60,160 calls with no rate-rule refusal is untouched (§9.5) |
| **Y2** no Levin leaf across a stationary point | **holds** | 0 Levin leaves listed. Window leaves listed: 1,682 (analytic, 1e-7), 526 (group, each rtol), as at 07b |
| **Y3** every phase with something to locate is located | **holds** | 23,993 calls scored, 0 violations. 784 set aside (392 group calls per rtol), each `zero_rate` with no sign change |
| **Y4** honest where referenced | **holds** (R2-28 as §9.5 predicted) | 312 scorings: 299 honest, 3 unconverged, 6 reported and not scored, 4 converged and dishonest, all four R2-28. **(a)** 07b's U4's 190: 181 honest; v04 unconverged (0.000156); 6 not scored (R2-27, R2-29, R3a-0-33666-a13, each at both rtols; stated error above min(abserr)/100; ratio ≤ 0.035); R2-28 and R2-28~o converged and dishonest at **1.1434**. **(b)** Part A's 122: R4 58 honest and 2 unconverged (`c-1e-8-3-13490-2` 0.0086, `c-1e-8-3-13615-3` 0.0083); R5 60 honest; R2-28's census call at both rtols converged and dishonest at **1.1434** (`c-1e-7-0-13757-5`, `c-1e-8-0-13757-5`). No other converged dishonest call. Largest honest ratio 0.805. (b)'s 122 verdicts equal Part A's (S) verdicts on the same calls |
| **Y5** the gate is consistent; the guard a backstop (gated solves) | **holds** | 0 node-partition sums on gated solves, 0 Clenshaw–Curtis regions with a Levin half, 0 guarded regions. Descriptive: 18,238 forced solves took the node-partition sum (07b: 18,238) |
| **Y6** 06b's 24 honest or unconverged | **holds** (as predicted) | 24 of 24 honest and converged; ratios 1.58×10⁻⁵–0.828 |
| **Y7** the pairs | **holds** (as predicted) | 2,484 of 2,484 pairs have r_g ≤ 1, 0 unpaired; largest 0.467. `3-14501` 0.00046, `3-35006` 0.00057 (07b 0.00046, 0.00057). 11 pairs have an unconverged member (07b: 14) |
| **Y8** the code is (S) | **holds** | 453 of 453 ids (448 distinct census calls) equal (S)'s replay records in all five outputs. The stop acts in 327 of them |
| **Y9** the chase ends | **holds** | **(i)** 0 chase calls. The census has 178 unconverged calls, all group (82 at 1e-7, 96 at 1e-8). 11 have no leaf at depth 20; in the other 167, not every depth-20 leaf lies below a refused leaf. **(ii)** Of 07b's 198 newly unconverged calls, exactly five are unconverged here: `c-1e-8-1-2229-4`, `c-1e-8-3-13490-2`, `c-1e-8-3-13490-3`, `c-1e-8-3-13615-2`, `c-1e-8-3-13615-3`. Each is untouched (stops 0), equal to 07b's call, and has no leaf at depth 20 |

#### Descriptive (§6 and §9.5; not scored)

- **Where the fix acts** (`num_chase_stops` > 0): 82,172 of 289,482 calls (28.4%).

  | rtol, kind | calls | acted | with a counted bisection | stops | stopped leaves | bisections counted |
  |---|---|---|---|---|---|---|
  | 1e-7 analytic | 107,416 | 34,395 | 89,981 | 37,241 | 247,001 | 554,650 |
  | 1e-7 group | 37,325 | 12,550 | 18,093 | 13,377 | 277,949 | 113,273 |
  | 1e-8 analytic | 107,416 | 23,264 | 88,453 | 26,905 | 182,952 | 502,641 |
  | 1e-8 group | 37,325 | 11,963 | 17,949 | 12,653 | 269,387 | 111,249 |
  | **all** | **289,482** | **82,172** | **214,476** | **90,176** | **977,289** | **1,281,813** |

  - **By band:** seam 18,290 of 149,482 calls acted (12%), mid 39,754 of 92,400 (43%), low 24,128 of 47,600 (51%).
  - **By k:** from 20% of calls (k = 10⁵) to 32% (k ≈ 9.6×10⁷).
  - **By 07b's class of the call** (the rate rules' refusals at 07b): range-refused 81,527 of 228,648; edge and range
    638 of 665; edge only 7 of 9; no refusal 0 of 60,160.
  - **Windows:** 1,447 of 7,300 calls with a window acted. **The 720:** 15,655 of 30,240.
  - Tables by rtol × kind × band and rtol × kind × k: `out/scores/summary.json` `where_the_fix_acts`.
- **The stopped leaves, by the stop's instrument** (977,289; the instrument's counts equal the result's on every call, 0
  errors, every stopped leaf returned):

  | rtol, kind | Clenshaw–Curtis untrusted / trusted | Levin untrusted / trusted |
  |---|---|---|
  | 1e-7 group | 98,250 / 77,695 | 62,331 / 39,673 |
  | 1e-7 analytic | 8 / 105 | 167,632 / 79,256 |
  | 1e-8 group | 95,287 / 75,341 | 60,117 / 38,642 |
  | 1e-8 analytic | 0 / 8 | 123,419 / 59,525 |

  - **By route:** Clenshaw–Curtis 346,694; Levin 630,595 (64.5%).
  - **By depth:**
    - Clenshaw–Curtis leaves lie at depths 3–17, most at 11–13 (72,781, 145,304, 82,134).
    - Levin leaves lie at depths 1–15, most at 3 and 4 (149,757 and 349,120).
  - **By r:**
    - r ≥ 1: 180,906 Clenshaw–Curtis and 88,693 Levin (group); 278,168 Levin (analytic).
    - r < 1/4: 44,868 Clenshaw–Curtis (group); 50,801 Levin (group); 69,622 Levin (analytic).
  - **Share of the call's abserr** (a stopped leaf's e′ as returned over the call's abserr), carried by one stopped leaf:
    - at least 0.1: 605 Clenshaw–Curtis and 5,358 Levin leaves on group calls; 113 Clenshaw–Curtis and 45,121 Levin leaves
      on analytic calls;
    - the largest single share is 0.99999: `h-1e-7-1-35361` analytic #13, a trusted Clenshaw–Curtis leaf at depth 3,
      r 0.684.
- **`converged` against 07b's census:**
  - **newly unconverged: none**, at either rtol, analytic or group, so no carrying stopped leaf is listed
    (`out/stop3/newly_unconverged.md` is empty);
  - **newly converged: 193**, all group calls the fix acts in: 97 at 1e-7, 96 at 1e-8.
- **`converged` against 07a's census** (from 07b's stop-3 call list):
  - newly unconverged: 5, the five of Y9 (ii), all group at 1e-8 (untouched here, as at 07b);
  - newly converged: 5, the group calls at 1e-8 that 07b converged.
- **Unconverged calls:** 07c 82 + 96 (1e-7 + 1e-8), all group; 07b 179 + 192; 07a 82 + 96.
- **The calls still unconverged with every depth-20 leaf below a refused leaf:** none (Y9 (i)).
- **Rows (units) with a leaf at depth 20:** 77 at 1e-7 and 83 at 1e-8 (07b 182 and 185; 07a 77 and 83).
- **Rows' `total_converged`:** 136 false → true, 0 true → false.
- **abserr against 07b's:**
  - untouched calls: equal.
  - acted group calls: median 1.000; p01 0.62–0.63; p99 2.7; maximum 7.6 (1e-7) and 8.1 (1e-8); above 2× in 302 and
    307.
  - acted analytic calls: median 1.00 (1e-7) and 1.02 (1e-8); 24 above 2× at 1e-7, the largest 242×.
    - The largest is `h-1e-7-1-14470` analytic #7 (J1), converged under both codes.
    - Its abserr is carried (share 0.998) by an untrusted stopped Levin leaf at depth 3, with r = 0.99986.
    - That leaf declares max(residual, δ₁)/(1 − r) = 2.46×10⁻²⁰, from δ₁ = 3.5×10⁻²⁴.
- **Values moved past the sum** of the two codes' abserrs:
  - against 07b's: **0 calls**;
  - against 07a's: 328 group calls (07b's census: 326).
- **Row totals against 07b's:** 8,872 rows differ; the move over 07b's `total_abserr` has median 2.1×10⁻⁸ and maximum
  9.9×10⁻⁵; 0 rows above 1.
- **Cost** (`out/stop3/stop3.json`):

  | | 07c | 07b | ratio |
  |---|---|---|---|
  | census wall | 43,080 s | 45,016 s | 0.957 |
  | against the projection | 0.995 of the central 43,312 s; 0.931 of the highest margin 46,285 s | | |
  | unit wall, 1e-7 | 49,405 s | 52,154 s | 0.947 |
  | unit wall, 1e-8 | 49,325 s | 54,667 s | 0.902 |
  | regions, all calls | 5,536,209 | 6,039,639 | 0.917 |
  | regions, 1e-7 analytic / group | | | 0.965 / 0.883 |
  | regions, 1e-8 analytic / group | | | 0.976 / 0.891 |
  | largest unit wall | 40.0 s | 40.1 s | |

  0 units were over budget.

#### Deviations since stop 3

19. **Scorers, each copied from 07b's census harness and changed where marked "prompt 07c":**
    - `score_07c.py` (from `score_07b.py`; adds the stop's aggregation `r07c`);
    - `score_v4_07c.py` and `score_refs_07c.py` (Y4 (a) and Y6, with 07b's census as the other code);
    - `xcheck_07c.py` (Y7, before = 07b);
    - `summarise_07c.py` (from `summarise_07b.py`).

    New: `score_refs_partA_07c.py` (Y4 (b)). IMPLEMENTATION CHOICE.
20. **Arithmetic:**
    - Y4 (a)'s ratios are 07b's scripts', in mpmath at 40 digits.
    - Y4 (b), every value move and every abserr ratio are in exact Fractions.
    - R2-28 is scored both ways and agrees: 1.1434356254583664 at 1e-7 in each.

    IMPLEMENTATION CHOICE.
21. **"Class" in the descriptive tables** is read as 07b's class of the call by the rate rules, beside kind, window, the
    720, band and k. IMPLEMENTATION CHOICE.
22. **Scoring ran in the session through `rec.sh`**, not as registered runs. The longest took about 2 minutes.
    IMPLEMENTATION CHOICE.
23. **`KEPT_ON_MAC_STUDIO.txt` keeps, by judgment and not by size, the raw worker streams and unit records** of the Y0
    and smoke runs (`emit.jsonl`, `units.jsonl` under `state/`) and the smoke dry run of the Y8 scorer. Their verdicts are
    in `out/`. IMPLEMENTATION CHOICE.

#### Verification performed (final)

`verification.md` has the account. In brief:
- **The trees:** h is at `cc3368ce`, clean with `--ignored`, no `__pycache__`, at every check and at the end. The main
  checkout is at `28ebaf07`, clean.
- **Every census worker:** module `d141a80b` from the tree, no variant, h clean at its SHA, no Ray, no foreign module
  (18,560 of 18,560 units' evidence; 16 workers).
- **The scratch copy:** EQUAL to its certification and to its before-hash, after the census and at the end. `shasum -c`
  against 06e's after-hashes OK (5). No `-wal` or `-shm` files.
- **07b's census and K0 runs, 07b's records and Part A's records:** before, at stop 2, after the census and at the end,
  every hash equal (`out/runs_hashed_{before,stop2,after,end}.txt`).
- **The registry:** 96 runs, all finished. No lock was taken.

#### Observations not acted on (final; proposed in `issues.md`)

1. **The stop acts mostly on Levin leaves, and on analytic calls.**
   - 630,595 of the 977,289 stopped leaves are Levin leaves, most at depth 3–4, 413,499 of them untrusted.
   - 57,659 of the 82,172 acted calls are analytic: 429,832 stopped Levin leaves.
   - §9.3's basis was D1's chase on group calls: "A Levin leaf is rarely at the noise level (both differences within
     4 ν_B in 2,460 of 15,445 Levin bisections)", and "The analytic calls have no chase in the sample". The census reaches
     far beyond that basis.
   - No referenced call is converged and dishonest but R2-28, and no analytic call is unconverged. But the stopped Levin
     leaves' declarations are scored only through the referenced analytic calls (07b's R2's 21 and R3's).
2. **A stopped untrusted leaf's declaration can overstate by the factor 1/(1 − r).** On `h-1e-7-1-14470` #7, r = 0.99986
   gives 2.46×10⁻²⁰ from δ₁ = 3.5×10⁻²⁴; the call's abserr grows 242× against 07b's, and it stays converged. This is the
   pattern the LiouvilleGreen draws showed, at census scale. A stopped leaf declares what the tree declares at
   `depth_max`, as §1.1 sets.
3. **The 324 calls of item 6 are unchanged in kind:** 328 group calls move past the sum of 07c's and 07a's abserrs (07b:
   326 against 07a). None moves past the sum against 07b. They remain 07d's.
4. **13 units untouched in every call** differ from 07b's whole record only in the harness's own post-section re-solves
   (`analytic_06h.parent_counterfactual`, `route_07a_post_unit.n_solves`), not census outputs (stop 3).
5. **Unconverged calls and rows at depth 20 return to 07a's counts** (82 + 96 calls; 77 / 83 rows). The 178 unconverged
   calls all predate 07b, and 167 of them have depth-20 leaves not all below a refused leaf.

#### State handed on

- **Nothing is running.** Both registered runs are `done`: `levin-phase-floor-07c-k0-20261119T151009` (11/11) and
  `levin-phase-floor-07c-census-20261119T201419` (18,560/18,560). No lock was taken.
- **Kept on the Mac Studio** (`KEPT_ON_MAC_STUDIO.txt`):
  - the census run directory in full (149 files, 10.7 GB): **prompt 07d's "before"**;
  - the k0 run directory;
  - every pickle, every file over 2 MB, and the raw worker streams under `census/`.
- **Read only and unchanged:** the scratch copy, 07b's census and K0 runs and records, and Part A's records, by hash.
- **Issues:** proposed in `issues.md`, for the laptop and the user.
- **No commit, no push.**

## levin-phase-floor 07c, Part A (section 2): the measurement — log of the measuring agent

**Subject:** prompt 07c Part A, the chase below a refused leaf: section 2.1's calls, section 2.2's D1, section 2.3's D2,
section 2.4's variants, section 2.5's references R4 and R5, and section 2.6's tables. Run on the Mac Studio (mps026298).
The worktree is `/Users/ds283/Documents/Code/SecondaryGWKit-lpf07c-2fd979f`, at the dispatch SHA
`2fd979fd95fa4d4ff3ac750e0c4094d83d744f7f`, whose code is 07b's (`5e19c4b6`, `AdaptiveLevin/levin_quadrature.py`
`8956c7b6…`). Records are in `/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07c/measurement/`.
**Commit:** none. The orchestrator commits the records, on the laptop.
**Result (2026-11-19): the planned stop of section 2.7.**
- **One stop of section 8's kind came first.** It was put to the orchestrator, ruled on, and is recorded as deviations 23
  and 24: the first references run took no reading in 12.4 h.
- **Three of the seven variants qualify:** N, Eg and Ea.
- **S, S2, SEg and SEa fail** on R2-28 alone. That call converges under them and is dishonest, at ratios 1.12–1.14.
- **Table 4:** chase calls left unconverged, of 201:

  | tree | S | S2 | N | Eg | Ea | SEg | SEa |
  |---|---|---|---|---|---|---|---|
  | 201 | 5 | 6 | 21 | 186 | 186 | 4 | 4 |
- **The gate** holds before and after (`out/gate/`).
- **Section 2.1:** every replay at 07b's code reproduces its record bit for bit: 454 of 454 (the 198 chase calls and
  their 8 counterparts at the other rtol, 07b Part A's 187, R31's record against R31's captured record, R5's 43 and
  their 17 counterparts). Captures: 198 of 198 and 43 of 43 equal 07b's census.
- **D1:**
  - The census agent's descent analysis is reproduced on all 198 calls. Every depth-20 leaf is below a refused leaf in
    192, and 6 have no depth-20 leaf.
  - 24,109 bisections below refused leaves are tabulated by route and cause: Levin 15,445, Clenshaw-Curtis 8,664.
  - The chase's 12,520 Clenshaw-Curtis leaves are re-solved with their true errors. 6,615 of them were bisected with a
    residual above their floor. On 2,968 of those, CC_47's true error is at or below the floor; on 1,159 it still is
    with the reference's own error added.
  - **Honesty with converged set aside:** on the 61 chase calls referenced (R4's 58, R2-28's 2, R31's record), the
    tree's |v − ref|/abserr is at most 0.416, except R2-28's two at 1.107.
- **D2:** no candidate estimate bounds CC_47's true error on every sampled leaf; none qualifies.
- **Variants:** seven run on all 454 calls (S, S2, N, Eg, Ea, SEg, SEa), each with its instrumented copy on the chase's
  259 calls, bit-identical to it.
- **References** (`levin-phase-floor-07c-refs2-20261118T235323`): 123 readings, each with exit 0, in 6 h 22 min.
  - They cover R4 (30 chase integrands), R5 (30 integrands, 60 calls) and R31's record.
  - Every call they serve meets the scoring rule: 121 of 121.
  - **R5:** all 43 calls dishonest at 07a's code are honest at 07b's, at ratios 0.00014–0.11. Their 17 counterparts are
    honest too, at ≤ 0.198.
- **Qualification** (`out/tables/scores_07c.md`):
  - N, Eg and Ea qualify.
  - S, S2, SEg and SEa do not. Each makes R2-28's four rows converged and dishonest. Those are two calls, each scored
    twice (deviation 21).
  - **Eg and Ea carry no bound basis:** D2 found that tail_sum4 does not qualify as an estimate (deviation 11).

### What shipped

Nothing in the tree. Its `git --no-optional-locks status --short --ignored` was empty at every check, and the main
checkout stayed at `28ebaf07`. The records, all under `measurement/`:
- **`scripts/`:**
  - 07b's census harness, copied whole from `var/levin-phase-floor-07b/census/scripts/` (230 files,
    `out/scripts_07b_census_sha256_{source,as_copied}.txt`). Changes are marked "prompt 07c": `prov.py`, `rec.sh` (written
    fresh from 07b Part A's `scripts/ext/rec.sh`, paths only) and `x1_capture_07c.py` (07b Part A's `x1_capture_07b.py`,
    its title).
  - 07b Part A's scripts, in `scripts/from_07b_measurement/` (276 files, byte-identical,
    `out/scripts_07b_measurement_sha256_{source,as_copied}.txt`).
  - The new scripts: `gate_hashes_07c.py`, `populations_07c.py`, `extract_census_refs_07c.py`,
    `extract_census_refs_extra_07c.py`, `capture_selection_07c.py`, `launch_07c.py` (from 07b's `launch_07b.py`),
    `make_calls_07c.py`, `replay_07c.py`, `run_replays_07c.py`, `plan_replays_07c.py`, `make_instr_07c.py`,
    `register_variant_07c.py`, `make_variants_07c.py`, `d1_descent_07c.py`, `leafcc_07c.py`, `leafcc_specs_07c.py`,
    `d2_table_07c.py`, `d1_tables_07c.py`, `d2_reproduce_07c.py`, `d2_population_07c.py`, `d2_regimes_07c.py`,
    `draw_refs_07c.py`, `slice_refs_07c.py`, `leafref_07c.py` (07b's `leafref`, its `TREE`), `refs_projection_07c.py`,
    `plan_refs_07c.py`, `plan_refs2_07c.py`, `variants_descent_07c.py`, `variant_reach_07c.py`, `score_07c.py` (v2). `run_06j.py` (07b's
    run driver) is changed to v2 where marked "prompt 07c" (deviation 24), and `leafref_07c.py` is v2 (deviation 23).
  - Superseded first forms in `scripts/superseded/`: `populations_07c.py` v1, `register_variant_07c.py` v1,
    `score_07c_v1.py`, `leafref_07c_v1.py`, `run_06j_v1.py` (deviations 3, 12, 20, 23, 24).
- **`scripts/variants/`:** the instrument `levin_quadrature_07c_instr.py` (`7c0a5b86…`, +83/−2 against the tree), and
  `levin_quadrature_07c_<v>.py` for S `09088683…`, S2 `b7819ee2…`, N `fbbdcbed…`, Eg `7d32bc1e…`, Ea `fd715e3d…`,
  SEg `50672dd7…`, SEa `33fedbdf…`, with their instrumented copies `<v>_i`. Each diff against the tree is in
  `out/variants/<v>.diff`, each SHA-256 in `out/variants/variants.json`, each basis in `state/variants/<v>.json`.
- **`state/`:** `seeds_07c.json` (`05f428f3…`, written 10:24:13Z, before any draw); `selection_r4.json`
  (`01918f2b…`), `selection_r5.json` (`e901adff…`), `selection_d2.json` (`c4beb666…`), each saved before any reading or
  solve; `refs_rule.txt`, `r4_rule.txt`, `r5_rule.txt`, written before any reading; `populations_07c.json`
  (`33910d71…`); `calls_07c.json` (`fcdbeffc…`, 394 calls) and `calls_07c_r5.json` (`7a469f20…`, 60); the census
  records (`census_refs_07c*.pkl`); the captures (`capture_chase/`, `capture_r5/`); the leaf specs (`leafcc_d1/`,
  `leafcc_d2/`); the reference inputs and specs (`refs/`); the plans with their purpose and scope texts; and
  `deviations_draft.md`.
- **`out/`:** `cmd/` and `commands.log` (every command and its output); `gate/`; `captures/`; `replays/` (tree,
  instrument, each variant and each `<v>_i`, with summaries); `d1/` (`d1_descent.json`, `d1_cc_leaves.{json,md}`,
  `d1_tables.{json,md}`, `variants_descent.json`); `leafcc_d1/`, `leafcc_d2/`; `d2/` (`reproduce.json`,
  `population.json`, `d2_table.{json,md}`, `d2_regimes.json`); `variants/` (`variants.json`, the diffs,
  `reach.{json,md}`); `refs/` (`inputs.json`, `refs_projection.json`, `readings/`); `tables/` (`scores_07c.{json,md}`);
  `cmd/refs/failed_first/` (the failed first references run's two job logs, 268 MB each, deviation 23); and the test
  outputs, which no table reads: `replays_test/`, `leafcc_test/`, `tables_test/`, `refs_test/`.
- **Registered runs** (`var/runs/`): `levin-phase-floor-07c-capture-chase-20261118T101845`,
  `-capture-r5-20261118T105127`, `-base-20261118T110242`, `-refs-20261118T112553`, `-variants-20261118T113152`,
  `-refs2-20261118T235323`. Every one has a recorded end. Five are done; `-refs-` was killed by its own backstop, with
  0 of 123 (deviation 23).

#### The variants (section 2.4), each with its basis, and the leaves it changes

The noise level is the leaf's own trust-check yardstick ν_B: a leaf is *at the noise level* if δ₁ ≤ C ν_B and δ₂ ≤ C ν_B.
**Basis** (`out/d1/d1_descent.json`): of the 8,664 Clenshaw-Curtis bisections below refused leaves (the 198 chase calls,
R2-28 at both rtols and R31's record), both δ ≤ ν_B in 1,288 (15%), ≤ 2ν_B in 6,342 (73%), ≤ 4ν_B in 7,665 (88%),
≤ 10ν_B in 8,428 (97%). Their δ₁/(φ_P + φ_L + φ_R) has median 0.68 and 90th percentile 1.01. D2 finds their CC_47 true
errors noise-dominated in 22,700 of the 22,876 chase leaves measured. Of the 15,445 Levin bisections, δ₁/ν_B has median
47, and both δ ≤ 4ν_B in 2,460. **Where the basis does not reach:** the analytic calls have no chase in section 2.1's
sample, and the limit 6 is dqage's iroff1 limit (07a's window stop at the same noise level), not measured here.

| variant | what it is | SHA-256 | constants and margin |
|---|---|---|---|
| **S** | (S): below a refused leaf, bisections at the noise level count against the subtree's root; at 6 the subtree stops; every leaf of it at the noise level, and every one made in it later once found so, is floor-limited by `_floor_limit_stopped_window_leaf` | `09088683…` | C = 4 (88% of the CC bisections; the 90th percentile of δ₁/φ is 1.01, so 4 is about 4× it); limit 6 |
| **S2** | S at C = 2 | `b7819ee2…` | C = 2 (73%), the sensitivity of C |
| **N** | (N): a checked leaf with δ₁ ≤ ν_B, δ₂ ≤ ν_B and r ≥ 1 trusted, declaring max(residual, δ₁ + δ₂) | `fbbdcbed…` | none added. D1 found 1,288 CC and 983 Levin such leaves |
| **Eg** | (E-gate): the C2 gate's CC leaves declare tail_sum4 in place of the pair | `7d32bc1e…` | none. **tail_sum4 does not qualify** (D2): flagged, no bound basis |
| **Ea** | (E-all): every CC solve, window pieces included | `fd715e3d…` | as Eg |
| **SEg** | S with Eg | `50672dd7…` | as S and Eg |
| **SEa** | S with Ea | `33fedbdf…` | as S and Ea |

**What each changes** (`out/variants/reach.md`, `out/cmd/41_window_changes_summary.txt`; leaves compared by span):

| variant | calls touched of 454 | leaves made (CC / Levin) | tree leaves not made (CC / Levin) | same span, record differs | windows changed: stop / leaves at stop / split count only |
|---|---|---|---|---|---|
| S | 328 | 2,787 / 1,280 | 13,418 / 3,517 | 4,388 / 2,036 | 0 / 0 / 41 |
| S2 | 311 | 2,817 / 935 | 11,670 / 2,347 | 5,724 / 2,161 | 0 / 0 / 31 |
| N | 438 | 1,548 / 855 | 6,280 / 1,984 | 1,695 / 2,410 | 0 / 7 / 68 |
| Eg | 394 | 20,549 / 155 | 4,838 / 112 | 19,543 / 0 | 0 / 5 / 74 |
| Ea | 404 | 21,459 / 155 | 5,786 / 112 | 20,924 / 0 | 1 / 107 / 14 |
| SEg | 405 | 14,052 / 1,475 | 15,568 / 3,696 | 8,813 / 2,009 | 0 / 5 / 77 |
| SEa | 411 | 14,962 / 1,475 | 16,516 / 3,696 | 10,194 / 2,009 | 1 / 107 / 14 |

"Split count only" is a window whose stop fell, with the same leaves, at a different `stopped_after_split`: the global
split counter moves when a variant changes the splits elsewhere. **Under Ea and SEa one window stop is lost:**
`a:R3w-0-2006-0~o`, window [0x1.7dd87bff0dd28p+4, 0x1.2aaf213520fc0p+5], stopped at split 27 with 17 leaves at the
tree, never stopped under (E-all) (iroff1 count 3).

**Untouched calls, bit-identical to the tree** (replay and region arrays' SHA-256; one per set in `reach.md`), for
example: S and S2 `c-1e-8-1-2229-4` (`da85f758…`); N `a:R2-02` (`ffc9e1ca…`); Eg, Ea, SEg and SEa `a:R2-00`
(`8afa1d28…`). N touches every chase call, and Eg, Ea, SEg and SEa every chase call too, so their untouched examples
come from other sets.

### Predictions and verdicts

Part A fixes no prediction (section 6's are Part C's). **The qualification rule** (section 2.6): a variant qualifies if,
on table 3, no referenced call is converged and dishonest that is not so at the tree, and, on table 4, it leaves fewer
chase calls unconverged than the tree.

| variant | new converged & dishonest (table 3) | chase unconverged of 201 (table 4) | qualifies | regions / tree, all 454 (table 5) | wall / tree |
|---|---|---|---|---|---|
| tree | – | 201 (195 with every depth-20 leaf below a refused leaf) | – | 1.000 | 1.000 |
| S | a:R2-28, a:R2-28~o, c-1e-7-0-13757-5, c-1e-8-0-13757-5 (ratios 1.143) | 5 (0) | no | 0.745 | 0.726 |
| S2 | the same four (1.120) | 6 (0) | no | 0.796 | 0.773 |
| N | none | 21 (19) | yes | 0.884 | 0.879 |
| Eg | none | 186 (182) | yes | 1.313 | 1.435 |
| Ea | none | 186 (182) | yes | 1.312 | 1.432 |
| SEg | the same four (1.139) | 4 (0) | no | 0.926 | 1.003 |
| SEa | the same four (1.139) | 4 (0) | no | 0.925 | 1.036 |

**Table 3** (305 referenced; 6 not scored, 07b's `a:R2-27`, `a:R2-29` and `a:R3a-0-33666-a13` at both rtols):

| variant | converged & honest | converged & dishonest | unconverged |
|---|---|---|---|
| tree | 235 | 0 | 64 |
| S | 292 | 4 | 3 |
| S2 | 292 | 4 | 3 |
| N | 285 | 0 | 14 |
| Eg | 228 | 0 | 71 |
| Ea | 228 | 0 | 71 |
| SEg | 287 | 4 | 8 |
| SEa | 287 | 4 | 8 |

Under Eg and Ea, seven calls that converge at the tree are unconverged; under SEg and SEa, five (`issues.md` item 8).
Each unconverged call's ratio is listed in `out/tables/scores_07c.md`. **R2-28** is off by the same −2.30×10⁻³⁷ under
every code (`issues.md` item 7). Table 5's "calls touched" uses a narrower definition than `reach.md` (`issues.md` item
10).

### Deviations from the prompt

1. **07b's census harness copied whole into `scripts/`** from `var/levin-phase-floor-07b/census/scripts/` (230 files,
   byte-identical: `out/scripts_07b_census_sha256_{source,as_copied}.txt`), all but its `rec.sh`; `rec.sh` is written
   fresh from 07b Part A's extension `scripts/ext/rec.sh` (paths only). 07b Part A's scripts are in
   `scripts/from_07b_measurement/` (276 files, byte-identical). `prov.py` is changed where marked "prompt 07c" (the tree,
   07b's census as read-only input). IMPLEMENTATION CHOICE.
2. **The copies were made with the shell's cwd in the records directory** (`cd` into it for `cp`), not the worktree, for
   one command. Nothing was written in the worktree. UNINTENDED DRIFT.
3. **`populations_07c.py`'s first run read 07b's comparison pairs as [07a, 07b]**; they are [07b, 07a] (07b's
   `compare_07b.py`: `[s, s7]`). It found 5 chase calls instead of 198. Its output is in `state/superseded/` and its
   first script in `scripts/superseded/`; the corrected run reproduces the census agent's 198 as a set, and 324 / 18 /
   306. UNINTENDED DRIFT (a failed first attempt).
4. **R31's record is compared with the record 07b's rehearsal agent captured** (`var/levin-phase-floor-07b/rehearsal/
   out/refcap.json`, which that agent showed to reproduce R31's store), not with R31's store: the brief forbids touching
   R31's store. It runs at its own rtol (1e-7) only: no record exists at 1e-8 to reproduce. STRUCTURALLY REQUIRED.
5. **The Clenshaw-Curtis pair is CC_N and CC_{2N-1} at the calls' order N = 24** (CC_24 and CC_47), not CC_16 and
   CC_31: the census calls run at `chebyshev_order` 24. The prompt's labels are the audit's (N = 16). STRUCTURALLY REQUIRED.
6. **The unit drawn for R4 and R5 is an integrand**, and every call of it in the population is scored from its one reading
   (R5: 30 integrands, 43 of the 306 calls). The rule is in `scripts/draw_refs_07c.py` and `state/r5_rule.txt`,
   written before the draw. IMPLEMENTATION CHOICE.
7. **Every integrand of section 2.1 runs at both rtols:** where a population holds one rtol only, the other is replayed
   from the same capture with the other census record's (rtol, atol) (07b Part A's deviation 8) and must reproduce 07b's
   census record there (`x-...` records, `extract_census_refs_extra_07c.py`). IMPLEMENTATION CHOICE.
8. **The leaf reference gained a noise-averaging rule after its first test.** On R31's record's 122 chase leaves every
   interpolatory rule (CC_93, CC_185, composite Gauss-Legendre) disagreed with every other at about 1e-37, above the leaf's
   floor: the integrand as evaluated carries noise (about 3-6e-12 of |g| per sample) the floor does not model. leafcc_07c.py
   now also fits a 600-point Chebyshev least-squares polynomial (degrees 24 and 32), and uses whichever reference states
   the smaller error. The first test's output is `out/leafcc_test/superseded/`. IMPLEMENTATION CHOICE.
9. **D2's population is the Clenshaw-Curtis leaves of section 2.1's calls**, not of the whole census: a leaf can be measured
   only with its call's integrand, and only those calls are captured (x1 capture takes group calls). Its analytic strata
   hold 13 leaves (6, 5 and 2), below the 20 a stratum the prompt asks for; there are no analytic window pieces and no
   guarded leaf. STRUCTURALLY REQUIRED.
10. **06d's numbers are not reproduced on their own case:** the records found name no case for "2.7e-6 against 5.4e-17 at
    6 pi". The audit's table (section 2.2, its own integrand at N = 13) is reproduced to the digits it prints; the same
    integrand at N = 16 gives 9.8e-8 at 6 pi and 3.8e-12 at 3 pi (out/d2/reproduce.json). STRUCTURALLY REQUIRED.
11. **(E) is run with tail_sum4, which does not qualify as an estimate** by section 2.3's rule (no candidate does; the
    smallest ratio to CC_47's true error is 0.20 on the gate leaves sampled). It is run, flagged, so that the tables show
    what a sharper estimate does; its variants (Eg, Ea, SEg, SEa) are reported but carry no bound basis.
    IMPLEMENTATION CHOICE.
12. **register_variant_07c.py's first form failed on 14 registrations** (its pattern wanted double quotes; the generator
    writes the name with repr's single quotes); v1 is in scripts/superseded/, the instrument's registration under it
    stands. UNINTENDED DRIFT (a failed attempt).
13. **The references run two readings at a time** (four processes), not 07b's three, so that D2 and the variants could run
    beside it in the other two slots; projected 17.7 h elapsed instead of 11.8 h. IMPLEMENTATION CHOICE.
14. **(S) is run at two noise factors,** C = 4 (S, with the combinations) and C = 2 (S2), for the sensitivity of the one
    constant D1's distribution does not fix sharply. Seven candidates in all, within eight; the instrument and the seven
    instrumented copies (for table 4's descent column) are not candidates. IMPLEMENTATION CHOICE.
15. **D2's leaves are classed noise- or truncation-dominated** (d2_regimes_07c.py: true error at most 3 sigma sqrt(sum w^2)),
    added after the first table showed every candidate below 1 somewhere. IMPLEMENTATION CHOICE.
16. **Test replays and a test leaf run** are kept in out/replays_test/ and out/leafcc_test/; no table reads them.
    IMPLEMENTATION CHOICE.
17. **Which leaves a variant changes is measured by comparing its replay with the tree's, leaf by leaf**
    (`variant_reach_07c.py`, `out/variants/reach.{json,md}`), not from the variant's own counters (`num_s07c_*`,
    `num_n07c_trusted`). The replay harness records the result's standard keys and the refusal counts, not a variant's
    added keys. The variants' designs say which leaves each may change; the comparison says which it did. IMPLEMENTATION
    CHOICE.
18. **The instrumented copies (`<v>_i`) run on the chase's 259 calls** (the 198 chase calls, the R2 set at both rtols,
    which holds R2-28, and R31's record), not all 454: table 4's descent column needs them only there. On all 259 each
    equals its plain variant bit for bit (`out/d1/variants_descent.json`, `_identity`). IMPLEMENTATION CHOICE.
19. **Every variant's window stops are compared against 07b's,** not only (E-all)'s (`out/cmd/41_window_changes_summary.txt`).
    Most changes are only the split count at which the same stop fell (`stopped_after_split`). IMPLEMENTATION CHOICE.
20. **score_07c.py was corrected once before it scored anything.** A dry run on the references then present (07b's, with
    none of this Part's readings taken) showed that the R5 verdict line could not format a call with no 07a record (an
    `x-` call at the other rtol). v1 is in `scripts/superseded/score_07c_v1.py`; the dry run's output is
    `out/tables_test/` (`out/cmd/42_score_dryrun.txt`), and no table reads it. IMPLEMENTATION CHOICE.
21. **R2-28's two calls are scored twice in table 3:** as 07b's item 2 calls `a:R2-28` (1e-7) and `a:R2-28~o` (1e-8), and
    as the chase calls `c-1e-7-0-13757-5` and `c-1e-8-0-13757-5`, whose replays are the same calls, bit for bit. Section
    2.6 lists 07b's 187 and the chase separately, so both are kept, and the duplication is stated with the table.
    IMPLEMENTATION CHOICE.
22. **The registered runs' driver runs with its cwd in the records directory** (`launch_07c.py`, from 07b's
    `launch_07b.py` unchanged in this), and the manifests record the main checkout's HEAD (`28ebaf07`). The launching
    shell's cwd was the worktree. Every job is given `PYTHONPATH` = the worktree, and every replay records the module it
    ran and its SHA-256. Nothing was written in the worktree. IMPLEMENTATION CHOICE (inherited from 07b's launcher).
23. **The first references run took no reading in 12.4 h.** `leafref_07c.py` v1 named its worker loader `leafref_07b`
    (07b's module name, kept from the copy), which is not on the reading's path, so every pool worker failed in its
    initializer and was respawned (190,485 `ModuleNotFoundError`s in each job log, `out/cmd/refs/failed_first/`, 268 MB
    each). The run's monitor stopped it at its backstop, at 23:48:26Z (`levin-phase-floor-07c-refs-20261118T112553`, state
    killed, 0 of 123). The script was never test-run before the launch. v2 names `leafref_07c`; v1 is in
    `scripts/superseded/leafref_07c_v1.py`. A smoke reading on the smallest integrand (`out/refs_test/`,
    `out/cmd/43_refs_smoke_fccsplit.txt`) took 105 s against 101 s projected. 13.7 h of Part A's 36 h were used when it was
    found. The orchestrator was asked how to relaunch. UNINTENDED DRIFT.
24. **The references were relaunched three at a time** (six worker processes; nothing else was running), on the laptop
    orchestrator's ruling (option (b)), as `levin-phase-floor-07c-refs2-20261118T235323` (`state/plan_refs2.json`
    `cd35f76b…`, from `plan_refs.json` by `plan_refs2_07c.py`). It reads the same 123 readings, with the same rule, seeds
    and projection model: 11.8 h projected with the 1.2 margin, against 17.7 h at two at a time. Its backstop is
    74,784 s: Part A's 36 h less 1.5 h, from the plan's writing. To make that backstop wall-clock, `run_06j.py` v2 (07b's
    driver, changed where marked "prompt 07c"; v1 in `scripts/superseded/run_06j_v1.py`) divides the remaining
    projections in P by the plan's `backstop_divisor` (3). v1's P sums them serially, so at three at a time it would
    have stopped the run at its start. IMPLEMENTATION CHOICE (on the orchestrator's ruling).

### Verification performed

See `verification.md`. In brief:
- the gate is equal before and after (records 01 and 47);
- the tree's status was empty at every check, and the main checkout stayed at `28ebaf07`;
- 454 of 454 replays reproduce 07b's census at the tree and at the instrument;
- each `<v>_i` equals its variant on 259 of 259;
- the captures are 198 of 198 and 43 of 43;
- every Result-line number was checked against `out/tables/scores_07c.json`.

### Observations not acted on

See `issues.md`, items 1–10, proposed and none opened:
- integrand noise above the CC floor;
- the CC abserr below its true error;
- window pieces' noise far above their floor;
- the chase mostly Levin bisection;
- 06d's case not found;
- the "nearest" column;
- R2-28's common error;
- (E)'s newly unconverged calls;
- the projection's overestimate;
- table 5's "touched" definition.

Also, as the brief said, `var/levin-phase-floor-07b/rehearsal/out/commands.log` is 1,402 bytes longer than its list
says. It was not acted on.

### State handed on

- **No process of this Part is running.** Every registered run has a recorded end. Ray and the rehearsal lock were
  never used.
- **The worktree** is unchanged at `2fd979fd`.
- **The records** are under `var/levin-phase-floor-07c/measurement/`.
  - The pickles and every file over 2 MB stay on the Mac Studio, listed with SHA-256 and size in
    `KEPT_ON_MAC_STUDIO.txt`, which includes the failed first run's logs in `out/cmd/refs/failed_first/`.
- **For Part B and the orchestrator:**
  - the tables, in `out/tables/scores_07c.{json,md}`;
  - the variants and their SHA-256s, in `scripts/variants/` and `out/variants/variants.json`;
  - this Part's references, in `out/refs/readings/`;
  - the seeds, in `state/seeds_07c.json`: R4 707304, R5 707305, D2 707302.

### Extension: R2-28

**Subject:** the laptop orchestrator's extension of Part A (2026-11-19), on the user's ruling at the stop: find the cause of
R2-28's error before the fix is chosen. It asks five questions, answered below in its order. The extension used the
same worktree at `2fd979fd` and the same records directory, with no Ray, no lock and no store. Its records are
48–53 in `out/cmd/`. Its scripts are new: `r2_28_spec_07c.py`, `r2_28_table_07c.py`, `r2_28_probe_07c.py`,
`r2_28_mech_07c.py` and `r2_28_summary_07c.py`. Its outputs are in `out/r2-28/` and its inputs in `state/r2-28/`.
**Commit:** none.
**Result (2026-11-19):**
- **One leaf carries R2-28's error, and every 07b-lineage code makes it.** The leaf is the trusted Levin leaf
  [22.3566866, 23.5393356], at depth 4: the call's last sixteenth.
  - Under every code it has the same span and the same hex value (`-0x1.b0e16966a7238p-96`).
  - Its error is −2.338×10⁻³⁷ against a reference stated to 2.9×10⁻⁴², which is 1.247 of its e′ (1.874×10⁻³⁷).
  - The rest of every partition errs by +3.5 to +3.7×10⁻³⁹: 0.017–0.018 of the call's abserr, and 0.16–0.27 of the
    abserr less this leaf's e′.
- **The mechanism is 07b's M2**, the error common to a trusted Levin leaf, its halves and its quarters:
  - the leaf's own Levin solve errs by −4.8×10⁻³⁸;
  - its halves' sum and its quarters' sum both err by −2.34×10⁻³⁷;
  - its eighths' sum errs by +1.0×10⁻³⁸.

  r = 0.0048 trusts the leaf, and the leaf returns its quarters' sum. e′ is built from δ₁ = |V_P − S1|, the difference of
  two errors of the same sign, so it is smaller than the quarters' error.
- **It is not the CC fallback, and not a leaf the rate rules made by refusing it.**
  - The CC leaves' errors sum to 7×10⁻⁴¹.
  - The leaf passed its own range rule (q = 3.26). Its one refused ancestor is the root, refused by the range rule at
    q = 5.76×10⁷; its depth 1–3 ancestors failed the trust test on r ≥ 1.
- **07a's one region:** its value errs by −4.82×10⁻³⁸, 0.21 of 07b's error.
  - The error sits in one quarter, the last, [18.81, 23.54], which errs by −4.82×10⁻³⁸. The other three quarters err
    by less than 4×10⁻⁴³ each. So there is no cancellation.
  - The value is honest because its e′ is 8.68×10⁻³⁴, the root's own floor, 18,000× its error. Its error is of the same Levin
    scatter, about 10⁻³⁷, not below it.
- **One call, not a class, as far as this Part's data reach.** R2-28 is the only referenced call on which every
  07b-lineage code errs above its abserr (4 rows: its two calls, each scored twice). Of R2-28's trusted Levin leaves,
  only the carrying leaf is above its e′ by more than 1.05×. The others above e′ are at 1.008–1.041, plus one (tree
  leaf 18) at the reference's own error level.
- **The size against the request:** R2-28's error, 2.30×10⁻³⁷, is 1.6×10⁻⁴ of its request. The request is atol,
  1.446×10⁻³³, which is above rtol·|I| at both rtols.

#### Question 1: where the error sits

**Method.**
- One reading of `leafref_07c.py` (record 49) over R2-28's whole span: split FCC at resolutions 1 and 2, split ratio 1.8
  (R2-28's call reference's method), 6 workers.
- Its points are every leaf end of a:R2-28 and a:R2-28~o under the tree and the seven variants, and of 07a's region,
  together with the halves', quarters' and eighths' break points of every trusted leaf of the tree's, S's and 07a's
  partitions of a:R2-28. That gives 1,999 points and 1,998 pieces (`state/r2-28/spec.json` `7336643e…`, record 48b).
- The reading took 81 s at r1 and 287 s at r2 (`out/r2-28/leafref.json` `79c0c9c1…`).
- A span's reference is the sum of its r2 pieces. Its stated error is √(Σ fit-noise variance) + |r2 − r1|, 07b's
  x1_table rule.
- The whole span's sum is −1.76168148866605873×10⁻²⁹, stated to 2.2×10⁻⁴¹. That is 8.3×10⁻⁴² from 07b's call
  reference.

**The table** (`out/r2-28/leaves.{json,md}`, record 50). In every partition the leaves' errors sum to the call's error,
−2.301×10⁻³⁷. By kind, in the tree's partition of a:R2-28:

| kind | n | sum of errors | sum of stated reference errors | sum of e′ |
|---|---|---|---|---|
| trusted Levin | 46 | −2.301×10⁻³⁷ | 1.2×10⁻⁴⁰ | 1.98×10⁻³⁷ |
| trusted direct (CC by the C2 gate) | 188 | −1.8×10⁻⁴² | 6.2×10⁻⁴² | 9.6×10⁻³⁹ |
| floor-limited direct | 44 | +7.1×10⁻⁴¹ | 4.5×10⁻⁴² | 4.0×10⁻⁴⁰ |
| floor-limited Levin | 16 | −2.3×10⁻⁴² | 1.9×10⁻⁴¹ | 2.7×10⁻⁴¹ |
| untrusted direct (depth 20) | 2 | −3.5×10⁻⁴⁵ | 3.5×10⁻⁴⁵ | 3.8×10⁻⁴⁴ |

**The leaves:**
- **The carrying leaf** is tree leaf 295 (S's 99).
  - It is a trusted Levin leaf at depth 4, [0x1.65b4fd067cd45p+4, 0x1.78a11e65a7f80p+4] = [22.3566866, 23.5393356],
    with value kind quarters.
  - Its error is −2.338×10⁻³⁷ against its reference −2.13426569258546191×10⁻²⁹, stated to 2.9×10⁻⁴²; its e′ is
    1.874×10⁻³⁷, so error/e′ = −1.247.
  - Its |θ′| runs from 372.7 to 1,216 (ratio 3.26), and its phase variation V is 844 rad.
  - **Its rule history** (from the instrumented replays, tree and S_i alike):
    - the root, [4.617, 23.539], passed the trust test (r 0.025) and was refused by the range rule, q = 5.76×10⁷;
    - its ancestors at depths 1, 2 and 3 ([14.08, 23.54], [18.81, 23.54], [21.17, 23.54]) were untrusted by the trust
      test, clause r ≥ 1 (r 18.1, 1.16 and 5.17);
    - the leaf itself passed the range rule (q 3.26) and was trusted by r < 1/4 (r 0.0048, δ₁/ν_B 2.7×10⁴);
    - no edge rule acted on R2-28 (0 edge refusals).
- **Its neighbour**, leaf 294 (depth 5, [21.765, 22.357], trusted Levin, |θ′| 1,216–2,197), errs by +3.71×10⁻³⁹, which
  is 0.42 of its e′. Every other leaf errs by at most 3.6×10⁻⁴¹.
- **The same leaf, with the same hex value, is in every 07b-lineage partition** (`out/r2-28/summary.md`, record 53):

  | code | regions | carrying leaf's e′ / call abserr | rest of the call: error | rest / abserr | rest / (abserr − e′) |
  |---|---|---|---|---|---|
  | tree | 296 | 0.902 | +3.70×10⁻³⁹ | 0.0178 | 0.181 |
  | S | 100 | 0.931 | +3.69×10⁻³⁹ | 0.0183 | 0.267 |
  | S2 | 133 | 0.912 | +3.69×10⁻³⁹ | 0.0180 | 0.205 |
  | N | 263 | 0.903 | +3.63×10⁻³⁹ | 0.0175 | 0.180 |
  | Eg, Ea | 359 | 0.897 | +3.53×10⁻³⁹ | 0.0169 | 0.165 |
  | SEg, SEa | 100 | 0.928 | +3.69×10⁻³⁹ | 0.0183 | 0.254 |

  The ~o calls give the same figures to the digits shown.

#### Question 2: why

**Method.** `r2_28_probe_07c.py` (record 51, `out/r2-28/probe.json`) re-solves the leaf, its neighbour and 07a's region
at the tree's module, which is imported unchanged. Each span is re-solved as P, its halves, its quarters and its eighths,
under four rules:
- the driver's solve;
- the forced Clenshaw-Curtis pair;
- Levin at N + 8 = 32;
- Levin at N − 8 = 16.

Each probe reproduces its replay's leaf by hex (V_P = `value_own`, Q = `value`). `r2_28_mech_07c.py` (record 52,
`out/r2-28/mech.{json,md}`) sets every piece against its own reference.

**The carrying leaf's sums**, each against P's reference:

| rule | V_P | S1 = L + R | Q (returned) | S3 (eighths) |
|---|---|---|---|---|
| driver (N = 24) | −4.82×10⁻³⁸ | −2.347×10⁻³⁷ | **−2.338×10⁻³⁷** | +1.01×10⁻³⁸ |
| Levin N + 8 | −1.50×10⁻³⁷ | −1.98×10⁻³⁷ | −2.04×10⁻³⁷ | −9.95×10⁻³⁸ |
| Levin N − 8 | −8.86×10⁻³⁸ | −7.12×10⁻³⁸ | −1.10×10⁻³⁷ | −2.58×10⁻³⁷ |
| forced CC | −6.1×10⁻²⁸ | +6.4×10⁻²⁸ | +3.6×10⁻²⁹ | −5.7×10⁻³⁰ |

**What separates the candidates:**
- **07b's M2, an error common to the leaf's halves and quarters, fits.**
  - S1 and Q carry the same error, −2.35 and −2.34×10⁻³⁷. That makes δ₂ = |S1 − Q| = 9.0×10⁻⁴⁰ and r = 0.0048, so the
    leaf is trusted.
  - The returned Q is wrong by 1.25 e′. e′ ≈ δ₁/(1 − r), and δ₁ = |V_P − S1| = 1.86×10⁻³⁷ is the difference between
    V_P's error (−4.8×10⁻³⁸) and S1's (−2.35×10⁻³⁷). Both have the same sign, so δ₁ is smaller than S1's error.
  - The eighths break the pattern (S3 +1.0×10⁻³⁸), as 07b found for M2.
  - Piece by piece, most of the error is in the right half R (−1.84×10⁻³⁷) and its right quarter RR = [23.244, 23.539]
    (−1.91×10⁻³⁷), at the call's upper end, where |θ′| is lowest. RR's eighths do not carry it: RRL −5.7×10⁻³⁹, RRR
    +7.8×10⁻³⁸.
  - δ₂ is small partly by cancellation between halves: L − (LL + LR) = −4.7×10⁻³⁸ and R − (RL + RR) = +4.7×10⁻³⁸.
- **The error is not truncation, and the floor does not model it.**
  - At the next orders the same pieces err by different amounts, at both N + 8 and N − 8. Over the leaf's 15 pieces
    and the three Levin rules (45 solves), the median |error| is 2.5×10⁻³⁸ and the largest 2.4×10⁻³⁷.
  - The median error is 2.0×10⁴ × the piece's floor (`phase_err`, 0.4–3.6×10⁻⁴²). The stated reference errors are at
    most 2.9×10⁻⁴², so the reference does not limit this.
  - On the neighbour, leaf 294, at |θ′| 1,216–2,197, the same measure is 8.4×10⁻⁴⁰, which is 2.3×10³ × the floor.
  - So the Levin solves on this span scatter at about 10⁻³⁷, or 5×10⁻⁹ of the leaf's value. That is above their
    floors, and does not fall with order or depth.
  - Its source within the solve (the integrand's noise, `issues.md` item 1, or the collocation) was not separated in
    this extension.
- **The CC fallback's estimate** is excluded: the carrying leaf is Levin. The CC leaves' errors sum to +7×10⁻⁴¹, against
  their e′ sum of 1.0×10⁻³⁸. The forced CC pair is no second rule here (V 59–844 rad, errors up to 10⁻²⁷); it shows
  only that CC cannot serve on these spans.
- **An edge or range rule's leaf**, in the sense of a leaf the rule made, is excluded:
  - there was no edge refusal;
  - the leaf passed its own range rule (q 3.26);
  - its depth 1–3 ancestors were untrusted by the trust test, not by a rate rule.

  The range rule's one part is the root's refusal (q 5.76×10⁷). That refusal sends the call into a partition at all;
  without it the call is 07a's one region (question 3).

#### Question 3: why 07a was right

07a's region is [4.617, 23.539], trusted at r 0.025, returning its quarters' sum Q. It is reproduced by the tree's module
by hex, both V_P and Q. The subregion solve is unchanged between `1ec734d7` and `5e19c4b6` (deviation 27).

| piece | span | driver's error | stated reference error | V |
|---|---|---|---|---|
| V_P (own) | whole | +2.77×10⁻³⁸ | 2.2×10⁻⁴¹ | 3.2×10¹⁰ |
| S1 | | −4.64×10⁻³⁸ | | |
| **Q (returned)** | | **−4.82×10⁻³⁸** | | |
| quarters LL, LR, RL | [4.617, 18.81] | −3.1, −2.5, −3.6×10⁻⁴³ | 1.4–5.9×10⁻⁴² | 4.7×10⁶–3.2×10¹⁰ |
| quarter RR | [18.81, 23.54] | **−4.82×10⁻³⁸** | 2.3×10⁻⁴¹ | 4.2×10⁴ |
| RR's eighths RRL, RRR | | −2.0×10⁻⁴², −8.14×10⁻³⁸ | | |

- **The value's error is in one piece, the last quarter. There is no cancellation:** the other quarters err by less than
  4×10⁻⁴³ each.
- That quarter is solved as one Levin solve over [18.81, 23.54], which contains the carrying leaf's span. It errs by
  −4.8×10⁻³⁸, a fifth of the error 07b's partition returns on [22.36, 23.54] alone.
- **07a is honest because its e′ is large, not because its value is exact.** Its e′, 8.68×10⁻³⁴, is the root's floor
  (`phase_err`, from V = 3.2×10¹⁰). That is 1.8×10⁴ × its error, and 0.60 of the request.
- Its value is more accurate than 07b's because one Levin solve happened to land closer. By the second rules on the same
  quarter, RR errs by −8.1×10⁻³⁸ at N + 8 and −5.2×10⁻³⁸ at N − 8. 07a's accuracy is of the same scatter, about 10⁻³⁷,
  not below it.

#### Question 4: one call or a class

From what this Part holds (`out/r2-28/summary.md`):
- **Calls.** Of the 305 referenced calls, including R31's record and the chase calls, 4 rows have every 07b-lineage
  code's error above its abserr. All four are R2-28: a:R2-28, a:R2-28~o, c-1e-7-0-13757-5 and c-1e-8-0-13757-5, which
  are two calls each scored twice.
- **Shared errors below the line.** Eight more rows show a shared error (the eight codes agree to a tenth of it) with
  every ratio above 0.5, none above 1:

  | call | rows | ratio |
  |---|---|---|
  | W2-2-14345-3 | 2 | 0.828 |
  | R2-08 | 2 | 0.783–0.805 |
  | R3a-3-843-a5 | 2 | 0.556 |
  | R2-00 | 2 | 0.542 |

  They were not attributed here.
- **Leaves.** This Part has references on trusted Levin leaves only for R2-28 (D1's and D2's are of CC leaves).
  - Of R2-28's 26–55 trusted Levin leaves in each partition, those above their e′ and resolved are:
    - the carrying leaf, at 1.247;
    - two at depths 11–12, at 1.008 and 1.041 (−3.4×10⁻⁴¹, −9.8×10⁻⁴²);
    - in the tree, N, Eg and Ea, one at depth 9 at 11.9× a tiny e′ (error 1.58×10⁻⁴², reference error 1.42×10⁻⁴²: at
      the reference's limit).
  - 07b's extension records the same kind elsewhere (07b `log.md`, observation 14): trusted Levin carriers on the pairs
    and v09, 1.17–2.66 e′ wrong, at 07b's variants. That was not re-measured here.
- **Count: one call (R2-28) in the scored calls; one carrying leaf above 1.05 e′ in the leaves read.**

#### Question 5: table 3, R2-28's rows annotated

Table 3 is unchanged (`out/tables/scores_07c.md`). Section 2.6's rule and the qualification are not re-scored.

| variant | converged & honest | converged & dishonest | unconverged |
|---|---|---|---|
| tree | 235 | 0 | 64, including R2-28's 4 rows at 1.107ᵃ |
| S | 292 | 4: R2-28's rows, 1.143ᵃ | 3 |
| S2 | 292 | 4: R2-28's rows, 1.120ᵃ | 3 |
| N | 285 | 0 | 14, including R2-28's 4 rows at 1.109ᵃ |
| Eg | 228 | 0 | 71, including R2-28's 4 rows at 1.102ᵃ |
| Ea | 228 | 0 | 71, including R2-28's 4 rows at 1.102ᵃ |
| SEg | 287 | 4: R2-28's rows, 1.139ᵃ | 8 |
| SEa | 287 | 4: R2-28's rows, 1.139ᵃ | 8 |

ᵃ **Cause, under every code:** the trusted Levin leaf [22.3566866, 23.5393356] at depth 4. It is the same span and the
same hex value under all eight codes, and it carries −2.338×10⁻³⁷ = 1.247 of its e′ by 07b's M2. The rest of each
code's partition errs by +3.5 to +3.7×10⁻³⁹, which is 0.16–0.27 of the code's abserr less that leaf's e′. The ratios
differ between codes only in the abserr each sums. No code's own change adds to the error.

#### Deviations (extension)

25. **A slip in `issues.md` item 7:** it says R2-28's reference was read "by fccsplit and QAWO at both resolutions".
    The scored reference has two readings, fccsplit r1 and r2 (`scores_07c.json`: basis `fccsplit-r1+fccsplit-r2`,
    spread 8.42×10⁻⁴²). The orchestrator found it. The earlier text is left as written, on the orchestrator's
    instruction. UNINTENDED DRIFT.
26. **`r2_28_spec_07c.py`'s first run (record 48) refused the ~o calls,** because their capture carries an rtol
    override. The blob and so the integrand are the same, so the reference is read from the 1e-7 capture, and the
    rerun (48b) checks the blob only. UNINTENDED DRIFT (a failed first attempt; nothing written).
27. **07a's region is read from 07b Part A's replay at 07a's code**
    (`var/levin-phase-floor-07b/measurement/out/replays/tree/R2-28.json`, read only). It is probed at the tree's module,
    because 07a's worktree no longer exists. The tree's module reproduces 07a's region (V_P and Q) by hex, and the
    tree's instrumented root check records the same own value and quarters' sum. 07a (`1ec734d7`) and 07b (`5e19c4b6`)
    differ by the rate rules' commit alone. STRUCTURALLY REQUIRED.
28. **One reference reading serves every partition.** The halves, quarters and eighths are read for the trusted leaves
    of the tree's, S's and 07a's partitions only, not the other variants'. The other variants' carrying leaf is the same
    leaf. IMPLEMENTATION CHOICE.
29. **The second rules are the forced Clenshaw-Curtis pair and Levin at N ± 8.** No rule independent of the Levin
    collocation and accurate at these phases was run on the pieces. The leaf references (split FCC) are the independent
    measure. IMPLEMENTATION CHOICE.
30. **Question 4's leaf-level count covers R2-28's partitions only.** This Part has no references on trusted Levin
    leaves elsewhere, and the orchestrator asked for no new population. STRUCTURALLY REQUIRED.

#### Verification (extension)

- **The worktree and the gate:** the worktree's git status was empty after the extension, and the main checkout stayed
  at `28ebaf07`.
- **The probes reproduce their leaves by hex:** each probe's driver solve reproduces its leaf, V_P and Q, on all three
  spans (`mech.md`).
- **The reading:** in every partition the leaf errors sum to the call's error against 07b's reference to four digits.
  The reading's whole span agrees with 07b's call reference to 8.3×10⁻⁴², within its stated 2.2×10⁻⁴¹.
- **The numbers:** every figure in this section's Result was checked against `leaves.json`, `mech.json` and
  `summary.json`.

#### Observations not acted on (extension)

`issues.md` items 11 and 12.

#### State handed on (extension)

- **Nothing is running.** The extension launched no registered run; its longest command took 6.1 min (record 49).
- **Records:**
  - the outputs are `out/r2-28/` (`leafref.json`, `leaves.{json,md}`, `probe.json`, `mech.{json,md}`,
    `summary.{json,md}`);
  - the inputs are `state/r2-28/` (`spec.json`, `probe_spec.json`);
  - the commands are records 48–53.
- **Kept on the Mac Studio:** `KEPT_ON_MAC_STUDIO.txt` gains one file in an appended section, `out/r2-28/leaves.json`
  (2,861,998 bytes).
