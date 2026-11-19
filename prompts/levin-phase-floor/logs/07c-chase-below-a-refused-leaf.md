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
