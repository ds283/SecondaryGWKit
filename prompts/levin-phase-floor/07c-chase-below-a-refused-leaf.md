# Prompt 07c — the chase below a refused leaf, and the Clenshaw–Curtis estimate

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-11-18, on the user's decisions at 07b's records commit (the board's Decisions,
**07b Part C**):
- on what follows 07b: *"Yes, items 6–8 as revised; write a 07c first to cover item 5."* Item 5 is
  `[07b-rate-range-rule-chases-noise-leaves-beyond-the-window-stop]`; item 6 is the 324 calls
  converged under 07a's and 07b's code whose values moved past both errors;
- on the $6\pi$ gate, asked *"What about the 6pi gate for sending a region to CC? Earlier in the
  campaign we noted this was under pressure from both sides?"*, and offered (a) a sharper
  Clenshaw–Curtis estimate measured here beside the stop-based candidates, with the threshold in a
  07d before 08; (b) one prompt for all; (c) this prompt on the chase alone: *"Go with (a); make the
  records commit, then write 07c"*.

Earlier, at 07b's extension stop: *"Go with (h), and record the chase as an issue"*.

**Where it runs:** in three parts, as 07b did.
- **Part A, §2, the measurement:** on the Mac Studio, Ray-free, with no lock, in a worktree at the
  dispatch SHA, whose code is 07b's. It measures the chase at census scale and the Clenshaw–Curtis
  estimate, scores candidate fixes, and takes two reference samples. **It ends at a planned stop:**
  the user chooses the fix with its tables in front of them, and the orchestrator writes it into §1
  and §3 as an amendment (§9), committed and pushed before Part B. Amended at that stop (§9): the fix
  is variant (S), after an extension of Part A on R2-28.
- **Part B, §1, §3 and §4, the code:** on the laptop, by a fresh-context subagent, in the main checkout
  at `HEAD`. One commit.
- **Part C, §5 and §6:** on the Mac Studio, after the code commit is pushed: **§5, rehearsal R32,**
  under the lock, against 07b's R31; **§6, the census,** Ray-free, on 06's certified scratch copy of
  A3 v2, against 07b's census as the "before". **This census is 07d's "before"**, in place of 07b's.

The records of Parts A and C are committed on the laptop afterwards: Part A's with §9's amendment,
Part C's in 07c's records commit.

**Closes, in its records commit, on the user's word:**
- `[07b-rate-range-rule-chases-noise-leaves-beyond-the-window-stop]`, if §6's census returns no call
  unconverged through a chase (every depth-20 leaf below a refused leaf) and §6's honesty predictions
  hold;
- `[06d-cc-nested-estimate-measures-the-coarse-rule]`, if the chosen fix includes a sharper
  Clenshaw–Curtis estimate and §6's predictions for it hold; otherwise it is narrowed by §2.3's
  measurement.

**Narrows, by its measurements:** `[07-c2-gate-threshold-was-never-measured]` (§2.3's table by phase
variation; the threshold itself is 07d's).

**Recommended model:** **Opus**, for every agent.

**Read first:**
1. **The board's §3 entries,** with every narrowing:
   `[07b-rate-range-rule-chases-noise-leaves-beyond-the-window-stop]`,
   `[06d-cc-nested-estimate-measures-the-coarse-rule]`, `[07-c2-gate-threshold-was-never-measured]`,
   `[06b-premise-check-can-chase-a-flat-residual-to-depth-max]` and
   `[06g-node-limited-stop-can-end-a-call-refinement-would-converge]`; and the Decisions rows **07b
   extension stop** and **07b Part C**.
2. **07b's prompt,** [`07b-levin-leaf-beside-a-window.md`](07b-levin-leaf-beside-a-window.md), §1, §2.7
   and §9; and 07a's §1.2 (the window's count, `dqage`'s `iroff1`).
3. **`AdaptiveLevin/levin_quadrature.py`:**
   - the comments above `SIX_PI`, `_LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO`, `_LEVIN_TRUST_MAX_RATIO`
     (its last paragraph, on the rate rules), the window's `iroff1` constants and
     `_LEVIN_EDGE_RATE_RATIO_MAX`;
   - `_adaptive_levin_subregion_cc` (CC₃₁'s value, $|{\rm CC}_{31} - {\rm CC}_{16}|$ its estimate, its
     `resasc`), `_cc_roundoff_floor`, `_cc_leaf_unresolved`;
   - `_trust_noise_yardstick`, `_leaf_rate_rule`, `_roundoff_exit_premise_holds`,
     `_window_bisection_counts`;
   - in `_adaptive_levin`: `_add_leaf` (how a stopped window reaches leaves made after the stop),
     `_check_leaf`, `_exempt_floor_limited`, `_floor_limit_stopped_window_leaf`, `_stop_window`,
     `_split_could_meet_budget`, and the main loop's step 1 (the budget and the lazy trust check), the
     round-off exit and the bisection with its window count.
4. **07b's Part A records,** `docs/levin-phase-floor/07b/measurement/`: `log.md` (the extension, X3,
   (i) and (i2)), `out/ext/x3/x3_chase.md`, `out/ext/tables/scores_ext.md`, `scripts/ext/`
   (`replay_ext.py` and the variants) and `scripts/variant_loader.py`; and its references (`out/refs/`, the rule
   in `scripts/score_07b.py`'s `refs()` and `ref_key()`).
5. **07b's census records,** `docs/levin-phase-floor/07b/census/`: `log.md`,
   `out/scores/summary.json` (`newly_unconverged`, `calls_moved_beyond_abserr_07a`),
   `out/scores/u4_u6_scores.json`, `out/scores/u4_v4_report.md`, and `scripts/`; and the Mac Studio
   review, `docs/levin-phase-floor/07b/review/review.md` and `final/moves.txt`.
6. **06d's measurement of the Clenshaw–Curtis pair,** `docs/levin-phase-floor/06e/investigations/07-leaves/out/tables.md`
   (the decomposition tables), and the 2026-09 audit `docs/adaptive-levin-audit-2026-09.md` §2.2.

**Gate (Part A):**
- 07b's records (`db72ca28`) and this prompt's commit are on `origin/handover-remedial`;
- the laptop's main checkout is clean, and `runregistrykit list` shows nothing `running` or stale on
  either machine;
- **the code is 07b's:** `git diff --stat 5e19c4b6 <dispatch SHA> -- AdaptiveLevin ComputeTargets
  LiouvilleGreen tools` is empty;
- on the Mac Studio: 07b's census checkpoint is `396784bc…`, and its run directory and
  `var/levin-phase-floor-07b/census/out/stop3/compare_census_calls.calls.jsonl` match the SHA-256 in
  07b's `census/KEPT_ON_MAC_STUDIO.txt`; R31's capture `ref-G+q+r-h-3-20363-1.pkl` and Part A's
  `ref-R2-28.pkl` match theirs; A3 v2's scratch copy matches its certification.

---

## 0. What this prompt is

### 0.1 The chase

07b's rate rules refuse a trusted Levin leaf whose phase rate it does not resolve. A refused leaf is
untrusted, and an untrusted leaf is refined whatever the budget says. Below it, the halves are
refined to the noise level, and there the trust check cannot settle:
- **The trust check at the noise level.** $\delta_1$ is about $10^{-43}$ against requests of about
  $10^{-33}$, and $r$ is anywhere from 0.3 to 8. The tree trusts a leaf if $r < 1/4$, or if $r < 1$ and
  $\delta_1 \le \nu_B$. With $r$ random, about half the noise-level leaves fail, and are bisected again.
- **The round-off exit is not reached.** In 07b's Part A (X3, under variant (f)), of 4,558 bisections
  below a refused leaf, 4,009 had the leaf's residual above its floor, so the exit was not considered,
  548 failed the exit's halving test, and one was bisected for the budget.
- **No stop counts these leaves.** 07a's window stop counts only a window's own leaves. Of (f)'s 4,374
  depth-20 leaves, 3,961 are Clenshaw–Curtis leaves of the C2 gate, 411 Levin leaves, and none a
  window piece.

So the call reaches the depth cap and is returned unconverged. **07b's census measured it in
production** (`docs/levin-phase-floor/07b/census/out/scores/summary.json`, `newly_unconverged`):
- 198 group calls converged at 07a are unconverged at 07b: 97 at `1e-7`, 101 at `1e-8`. They are 103
  distinct integrands (row and group index), 95 of them at both rtols; every group label and every $k$
  occurs.
- In 192 of the 198, every depth-20 leaf (668 in all) is a Clenshaw–Curtis leaf below a leaf the rate
  rules refused, at refused depths 0 (129), 1 (31), 2 (30) and 3 (2). The other 6 reach no depth 20 and
  miss their budget, with abserr 1.001–1.33× 07a's.
- Rows at depth 20 went from 77 and 83 to 182 and 185; 139 rows lost `total_converged`.
- **R2-28,** honest at 07a in one region (5.55e-05 of its abserr), ends at 296 regions, unconverged, at
  1.107 of its abserr: the only referenced call 07b made less accurate. **R31** has one such record:
  `G+q+r` at $(10^5, 10^6, 10^6)$, $z = 144627.7$, 2 → 143 regions.

These calls are not known to be wrong. Under (f), 59 of 61 such calls were within their abserr with
`converged` set aside. But they are unconverged, they cost, and they fill the depth-cap question
that 08 asks.

### 0.2 Why the Clenshaw–Curtis estimate

The fallback returns CC₃₁ and declares $|{\rm CC}_{31} - {\rm CC}_{16}|$, which is CC₁₆'s error
(`[06d-cc-nested-estimate-measures-the-coarse-rule]`). At a $6\pi$ phase span CC₁₆'s error is
$2.7\times10^{-6}$ against CC₃₁'s $5.4\times10^{-17}$; at $3\pi$, $1.4\times10^{-10}$ against
$3.4\times10^{-17}$. Two things follow:
- **The chase.** Most of its depth-20 leaves are Clenshaw–Curtis leaves whose residual sits above
  their floor, so the round-off exit, which would end their refinement, is not considered. If the
  residual were CC₃₁'s own error, many would be at their floor.
- **The gate.** Both pressures on $6\pi$ run through the estimate: above the gate it forces refinement
  where CC₃₁ is accurate, and just below it at tight tolerances Clenshaw–Curtis's summed declarations
  exceed the request (07a's three LiouvilleGreen draw calls). The threshold is 07d's; the estimate is
  measured here because it may be the chase's fix.

A sharper estimate touches every Clenshaw–Curtis leaf in the census, not only the chase's. It must
bound CC₃₁'s true error with a stated margin, on a measured sample covering the leaves it changes.

### 0.3 What 07b's (i) and (i2) taught

07b's extension tried two stops, each counting a refused leaf's subtree by 07a's window rule:
- **(i)** counted the whole subtree. Where the refused leaf is a seed, the count reached 6 after 9–11
  bisections of Levin leaves **not** at the noise level, and the stop returned them at $V_P$ with the
  $r$-based declaration the rate rules exist to refuse: `W2-14345` 1.55, `R1-x241-7` 2.44, dishonest.
- **(i2)** counted only the subtree's fallback leaves. It left none dishonest, but 59 calls still
  unconverged, because its stop, unlike a window's, does not reach the fallback leaves made after it
  by the subtree's Levin leaves.

So a stop must apply only to leaves at the noise level, and must reach every such leaf made after it,
as `_add_leaf` does for a stopped window.

### 0.4 The 324 calls (item 6)

07b's census has 326 group calls whose values moved from 07a's by more than the two codes' abserrs
summed, 324 of them converged under both (`docs/levin-phase-floor/07b/review/final/moves.txt`). In each,
one code's declaration fails. **18 are referenced, and on all 18 07b is honest (at most 0.054) and 07a
is not (1.37–10.8).** With R31's seven records, 25 of 25 referenced cases put the failure at 07a. But
those 25 were chosen for having moved or for 07a's failures. **The other 306 are unreferenced**, and
the user ruled that the next census takes a reference sample of them (§2.5, R5). It decides whether
07b's code, which is production's now, understates anywhere among them.

### 0.5 What it does not do

- **The $6\pi$ threshold** (`[07-c2-gate-threshold-was-never-measured]`): 07d's. §2.3's table by phase
  variation is evidence for it. No variant here changes `SIX_PI`.
- **The depth cap** (G2's L2): 08's experiment, after 07d.
- **07b's rate rules and their constants:** unchanged. A variant that changes which leaves they refuse
  is out of scope.
- `[07b-large-raw-phase-calls-can-be-marginally-dishonest]` and
  `[09-abserr-does-not-bound-phase-spline-floor]`: not this prompt's.

## 1. The code (Part B)

**Written at §2's stop, as §9's amendment,** from the user's choice. Whatever is chosen, §9 states:
1. **What changes,** in `AdaptiveLevin/levin_quadrature.py` only unless §9 says otherwise, with its
   reference implementation (Part A's variant) and its diff against the tree.
2. **Which leaves it touches,** and so which calls are predicted bit-identical to 07b's (§6 Y1).
3. **Its basis:** §2's measurement, with every constant named and commented, and where its basis does
   not reach a class of the census.
4. **What it does to `converged` and to the declarations.** No leaf is returned trusted, or stopped,
   on a declaration that §2's measurement shows can understate. A stopped leaf declares what it
   carries, as a stopped window's leaf does (`_floor_limit_stopped_window_leaf`).
5. **Diagnostics:** counts in the result dict for what the fix does, so that §6 can tell a call it
   acts on from one it does not.
6. **Docstrings and comments** that describe the trust check, the round-off exit, the fallback's
   estimate or the window's stop, rewritten to say what is now done and why, each listed by the agent
   with its line.

**As amended at §2.7's stop (§9), the fix is variant (S).** The user: *"I would choose S and open
`R2-28` as an issue that is othogonal to the fix that S is providing"*. Every change is in
`AdaptiveLevin/levin_quadrature.py`. The reference implementation is (S)'s variant,
`docs/levin-phase-floor/07c/measurement/scripts/variants/levin_quadrature_07c_S.py` (SHA-256
`09088683…`), with its diff against the tree, `out/variants/S.diff` (`ef5a6e9e…`). Every line the
diff adds is marked `# 07c variant (S)`; the names below replace the variant's.

### 1.1 The stop

Inside `_adaptive_levin`, as (S) does it:
- **At the noise level.** A leaf is at the noise level if its trust check one level down found
  $\delta_1 \le C\,\nu_B$ and $\delta_2 \le C\,\nu_B$, with $\nu_B$ the yardstick that check used
  (`_trust_noise_yardstick`'s value in `_check_leaf`) and $C$ = `_LEVIN_CHASE_NOISE_FACTOR`. A leaf
  not yet checked is not at the noise level. The test is a module-level function of
  $(\delta_1, \delta_2, \nu_B)$, so that §3 can test it directly.
- **The subtree's root.** In `_check_leaf`, a leaf a rate rule refuses (07b's `_leaf_rate_rule`, edge
  or range) roots a subtree, **unless it already belongs to one**: the root is the outermost refused
  leaf. At each bisection both children take their parent's root.
- **The count.** At each bisection of a leaf that has a root and is at the noise level, the root's
  count rises by one. When it reaches `_LEVIN_CHASE_STOP_LIMIT` the root stops, once: every current
  leaf with that root that is at the noise level is stopped.
- **Its reach.** At the end of `_check_leaf`, a leaf whose root has stopped and that is at the noise
  level is stopped. So a leaf made after the stop is stopped once its own check finds it at the noise
  level, as `_add_leaf` stops a stopped window's later leaves. **A leaf of the subtree not at the
  noise level is not stopped:** it is refined as before (07b's (i), §0.3).
- **A stopped leaf** goes through `_floor_limit_stopped_window_leaf`, unchanged, unless it is already
  floor-limited. It is returned floor-limited, exempt from the trust check, with its own value $V_P$;
  an untrusted one declares as an untrusted leaf at `depth_max` declares,
  $\max({\rm residual}, \delta_1)/(1-r)$ if $r < 1$ and $\max({\rm residual}, \delta_1 + \delta_2)$ if
  not; a trusted one keeps its widened declaration.
- **Unchanged:** the trust test, 07b's rate rules and their constants, the windows and their stop, the
  C2 gate, the round-off exit, `_exempt_floor_limited`, the fallback's estimate. A call in which no
  root reaches the limit is bit-identical to the tree's: the count alone changes nothing.

**The code reproduces (S)'s outputs bit for bit** (§9.5, Y8); a change that would alter any output
relative to (S)'s replays is a stop.

### 1.2 The constants

Each is a module constant with a comment stating its basis, its margin and where the basis does not
reach (§9.3).
- `_LEVIN_CHASE_NOISE_FACTOR = 4.0`, (S)'s $C$.
- `_LEVIN_CHASE_STOP_LIMIT = 6`, (S)'s limit.

### 1.3 What it does to `converged` and to the declarations

The tree's rule stands: a call is converged only if its budget is met, no returned leaf is untrusted,
and no guarded leaf is at the depth cap. A stopped leaf is floor-limited, not untrusted, so its
declaration counts against the budget as a stopped window's leaf does. A stopped leaf declares
exactly what the tree would declare for it at `depth_max` (§1.1). §9.3 states what §2 measured of
those declarations, and where it does not reach.

### 1.4 Diagnostics

The result dict gains three counts, over the whole run of the call: `num_chase_stops` (roots stopped),
`num_chase_stopped_leaves` (leaves stopped, at the stop and after it) and
`num_chase_noise_bisections` (bisections counted). They are (S)'s `num_s07c_stops`,
`num_s07c_stopped_leaves` and `num_s07c_counted`. **The fix acts in a call** if `num_chase_stops` is
positive. Nothing else in the result changes.

### 1.5 Docstrings and comments

Rewritten to say what is now done and why, each listed by the agent with its line: the module
docstring's item on the trust check and the rate rules; the comment block above
`_LEVIN_EDGE_RATE_RATIO_MAX` where it names the chase
(`[07b-rate-range-rule-chases-noise-leaves-beyond-the-window-stop]`); `_adaptive_levin`'s docstring
where it says a call can be chased to `depth_max`; `_check_leaf`'s docstring;
`_floor_limit_stopped_window_leaf`'s docstring, which now also stops a chase's leaf; and
`adaptive_levin_sincos`'s docstring on the three new counts.

## 2. The measurement — Part A, on the Mac Studio, a planned stop

**Run at the dispatch SHA, whose code is 07b's.** Nothing in the tree changes. Variants are copies of
`levin_quadrature.py` in the agent's `scripts/variants/`, loaded by 07b's `variant_loader.py`, each
with its diff against the tree saved. Records go in `var/levin-phase-floor-07c/measurement/`, in
`scripts/`, `state/` and `out/`, every output carrying the SHA-256 of its script and of the module or
variant it ran.

### 2.1 The calls

1. **The chase's 103 integrands,** the 198 calls of 07b's `newly_unconverged`, captured afresh from
   A3 v2's scratch copy by 07a's `make_capture.py` (as 07b's Part A captured), at both rtols where the
   census has both.
2. **R2-28** (07b's Part A capture, `state/refs/inputs/ref-R2-28.pkl`) and **R31's record**
   (`var/levin-phase-floor-07b/rehearsal/state/inputs/ref-G+q+r-h-3-20363-1.pkl`), each with its
   reference.
3. **07b's Part A calls,** the 187 of `census_wanted_07b.json` with their captures and references, as
   the regression set: every call there honest at 07b must stay honest or become unconverged.
4. **R5's calls** (§2.5), captured as item 1.

**Every replay at 07b's code reproduces 07b's census record bit for bit** (value, abserr, `converged`,
regions, depth, and the two refusal counts); one that does not is a stop. R31's record reproduces
R31's store values.

### 2.2 The chase, measured (D1)

On item 1's and item 2's calls, at 07b's code, with an instrument (a variant that records and changes
nothing, shown bit-identical to the tree on every call):
- **The census's descent analysis, checked independently:** for each call, its depth-20 leaves, their
  route (Levin, the C2 gate's fallback, a window piece, a guarded leaf), and whether each descends from
  a refused leaf. 07b's census agent found 192 of 198; the Mac Studio review did not re-check it.
- **Every bisection below a refused leaf:** the leaf's route and depth; $\delta_1$, $\delta_2$, $r$
  and $\nu_B$; which clause failed; its residual against its floor; if the round-off exit was
  considered, which of its tests refused it (the halving test, the budget clause); and the leaf's
  value move against the call's request.
- **For the chase's Clenshaw–Curtis leaves:** CC₃₁'s true error on the leaf's span (§2.3's reference
  rule) against $|{\rm CC}_{31} - {\rm CC}_{16}|$ and against the floor. How many of the 4,009-type
  refusals would pass the residual test with the true error in place of the estimate.
- **The six calls with no depth-20 leaf:** why they miss their budget, leaf by leaf.
- **Honesty with `converged` set aside,** against item 2's and R4's references (§2.5): which chase calls
  are within their abserr now.

### 2.3 The Clenshaw–Curtis estimate (D2)

A sample of the census's Clenshaw–Curtis leaves, drawn with a recorded seed before any is solved:
- **strata:** gate leaves by $V/6\pi$ (below 1/4, 1/4–1/2, 1/2–3/4, 3/4–1), the chase's leaves, window
  pieces, and guarded leaves at the cap if any; group and analytic calls; both rtols; at least 20 a
  stratum where the census has them;
- **for each leaf:** CC₁₆, CC₃₁ and a reference on the leaf's span by a rule independent of the pair
  (for example Clenshaw–Curtis on 4N−3 and 8N−7 nodes, or composite Gauss–Legendre, each with its own
  agreement stated), CC₃₁'s true error, $|{\rm CC}_{31} - {\rm CC}_{16}|$, the floor
  (`_cc_roundoff_floor`), `resasc`, and every candidate estimate;
- **candidate estimates, each with its basis and its cost in evaluations:** at least the tail of
  CC₃₁'s Chebyshev coefficients (the size of its last few, as Clenshaw and Curtis and later practice
  read it) and a third nested rule (CC on $4N-3$ nodes, whose difference from CC₃₁ measures CC₃₁'s
  error at the cost of $2N-2$ more evaluations). Others are the agent's, each with its basis.
- **The table:** for each candidate, the ratio of the estimate to CC₃₁'s true error over the sample,
  by stratum: its minimum (an estimate below the true error anywhere is not a bound), median and
  maximum, and how often it sits at or below the floor.

A candidate that understates CC₃₁'s true error on any sampled leaf by more than its stated margin
does not qualify as an estimate. 06d's numbers and the audit's are the agent's to reproduce first, on
their own cases.

### 2.4 The variants

At most eight. Each is a copy of the module with its diff and SHA-256 saved; each states which leaves
it changes, and shows a call it does not touch bit-identical to the tree. **Each constant has a stated
basis in §2.2's or §2.3's measurements, with its margin; where that basis does not reach a class of the
census, the variant says so.**
- **(S) A stop that reaches the leaves made after it.** Below a refused leaf, a count of bisections at
  the noise level (the agent defines "at the noise level" from §2.2's records, with its basis), and at
  a limit a stop that floor-limits every leaf of the subtree at the noise level and every such leaf
  made in it later, as `_add_leaf` does for a stopped window. A leaf not at the noise level is not
  stopped (§0.3's lesson from (i)). A stopped leaf declares as `_floor_limit_stopped_window_leaf`
  declares.
- **(E) The sharper estimate,** the best qualifying candidate of §2.3, as the fallback's `abserr`, in
  two scopes: **(E-gate)** on the C2 gate's leaves only, and **(E-all)** on every Clenshaw–Curtis leaf,
  window pieces included. A change to a window piece's estimate changes its window's count, so (E-all)
  reports every window whose stop changes against 07b's.
- **(N) The noise clause without $r$,** if §2.2 shows chase leaves with $\delta_1$ and $\delta_2$ both
  at most $\nu_B$ and $r \ge 1$: such a leaf trusted, declaring $\max({\rm residual}, \delta_1 +
  \delta_2)$ without the geometric factor. Not run if §2.2 shows no such leaf, with the count.
- **Combinations,** (S) with (E-gate) and (S) with (E-all) at least; the rest at the agent's choice.

Every variant is run on every call of §2.1, at both rtols.

### 2.5 The references

Read once, by 07's V4 method as 07b's Part A read them: `tools/filon_reference.py`, FCC at two
resolutions, QAWO where projected at 30 minutes or less, `fccsplit` where a segment holds a stationary
point, and Part A's rule for the reference and its stated error. **The rule is fixed in `state/` before
any reading.** A call is scored if the reference's stated error is at most min(abserr)/100 over the
codes and variants scored; otherwise reported, not scored.
- **(R4) The chase:** about 30 of §2.1 item 1's 103 integrands, drawn with a recorded seed, stratified
  by group label, $k$ and the refused ancestor's depth, both rtols scored from one reading. With R2-28
  and R31's record, these score every variant that returns a chase call converged.
- **(R5) Item 6:** about 30 of the 306 unreferenced calls of the 324 (converged under both codes, moved
  by more than the two abserrs summed: the rule of `review/final/moves.txt`, applied to
  `compare_census_calls.calls.jsonl`), drawn with a recorded seed, stratified by group label, $k$ and
  which code declares the smaller abserr. Each is scored at 07a's code (from 07a's census), at 07b's,
  and under every variant.

**The projection comes first,** by 07b's model with its $\gamma D_f$ term for every segment holding a
stationary point. Beyond 36 hours in all for Part A is a stop, reported before anything long runs,
with what would be dropped to fit.

### 2.6 The tables, fixed before the run

1. **(D1)'s tables:** the descent analysis against the census's; the bisections below refused leaves
   by cause; the chase's Clenshaw–Curtis leaves' true errors.
2. **(D2)'s table** (§2.3).
3. **Honesty:** for the tree and each variant, every referenced call of §2.1 (07b's 187, R4, R5, R2-28,
   R31's record) scored by 07's rule: converged and honest, converged and dishonest, unconverged (with
   $|v - {\rm ref}|/{\rm abserr}$).
4. **The chase:** for the tree and each variant, on §2.1 item 1's 198 calls and item 2's: how many are
   unconverged, how many of those end with every depth-20 leaf below a refused leaf, their regions and
   wall.
5. **Cost and reach:** for each variant, regions and wall on all of §2.1's calls against the tree's,
   and the calls it touches.

**A variant qualifies** if, on table 3, no referenced call is converged and dishonest that is not so at
the tree, and, on table 4, it leaves fewer chase calls unconverged than the tree. R5's verdicts at
07a's and 07b's code are reported beside the table, not folded into it.

### 2.7 Stop and report

At the end the agent reports §2.6's tables, which variants qualify, what (D1) and (D2) found, and R5's
verdicts, and stops. **It recommends nothing.** The user chooses at the stop, and the orchestrator writes
§1, §3 and §9.

## 3. Tests (Part B)

Written with §1 at §2's stop (§9). They include, whatever the fix:
1. **A synthetic call that reproduces the chase:** a refused leaf whose subtree reaches the noise
   level, unconverged at the depth cap at the parent, converged at the commit, honest against a
   closed-form or independently computed answer, with the fix's counts positive.
2. **The stop's reach,** if the fix has a stop: a leaf made after the stop is stopped too; a leaf not at
   the noise level is not.
3. **The estimate, directly,** if the fix has one: on constructed integrands with known CC₃₁ error,
   the estimate at or above the true error, within its margin.
4. **A call the fix must not touch,** bit-identical at the parent and the commit, its counts zero.
5. **Deliberate breakages,** each caught by the test named for it, each re-applied and reverted.
6. **Every changed existing assertion listed,** with why.
7. **The LiouvilleGreen draws** (07b's `docs/levin-phase-floor/07b/code/lg_draws_07b.py`, run at the
   parent and the commit): no test newly failing, and every call the fix should not touch
   bit-identical.

**As amended for (S) (§9).** In `AdaptiveLevin/tests/`, in a module the agent names. Item 3 does not
apply: (S) has no estimate.
- **(a) The noise level, directly,** on §1.1's function: $\delta_1 = \delta_2 = C\nu_B$ is at the noise
  level; either just above is not; $\nu_B = 0$ with both differences 0 is, and with either positive is
  not.
- **(b) The chase (item 1).** A synthetic call in which a rate rule refuses a leaf whose subtree
  reaches the noise level: unconverged at the depth cap at the parent, with every depth-20 leaf below
  the refused leaf; converged at the commit; honest against a closed-form or independently computed
  answer; `num_chase_stops` and `num_chase_stopped_leaves` positive. The chase needs integrand noise
  the floor does not model (Part A's `issues.md` items 1 and 12): a starting point is 07b's range
  mechanism in `test_levin_rate_rules.py` (c), with the integrand perturbed by a deterministic noise
  of stated size, the answer's error from that noise bounded and stated. A shallow `depth_max` is
  allowed if the call keeps the chase's shape.
- **(c) The stop's reach (item 2).** On (b)'s call or another, instrumented by the test only through
  the result's regions and counts:
  - a leaf made after its root stopped, at the noise level, is returned floor-limited;
  - a leaf of a stopped subtree not at the noise level is not stopped, and is refined or returned as
    at the parent;
  - a leaf at the noise level below a leaf the trust test refused, with no rate-rule refusal above
    it, is not counted and not stopped.
- **(d) The outermost root.** A refused leaf below a refused leaf: its bisections count against the
  outer root, and the stop is the outer root's.
- **(e) The declarations.** A stopped leaf that was untrusted declares
  $\max({\rm residual}, \delta_1)/(1-r)$ (or the $r \ge 1$ form) and returns $V_P$; one that was
  trusted keeps its widened declaration; each equals what the parent declares for the same leaf at
  `depth_max`.
- **(f) Not touched (item 4).** A call with no rate-rule refusal, and a call with refusals whose
  subtrees never reach the limit: bit-identical at the parent and the commit, `num_chase_stops` 0.

**If no call for (b) is found unconverged by a chase at the parent and converged and honest at the
commit,** the agent stops and asks, with the search it made (§8).

**Breakages,** each recorded with the test that catches it, then reverted:
- **(i)** `_LEVIN_CHASE_STOP_LIMIT` set to infinity: (b) and (c) must fail;
- **(ii)** `_LEVIN_CHASE_NOISE_FACTOR` set to infinity (every checked leaf at the noise level): (a)
  and (c) must fail;
- **(iii)** the stop not reaching leaves made after it: (c) must fail;
- **(iv)** children not taking their parent's root: (b) or (c) must fail;
- **(v)** the innermost refused leaf as the root: (d) must fail;
- **(vi)** bisections counted below any untrusted leaf, not only a refused one: (c) or (f) must fail;
- **(vii)** a stopped untrusted leaf keeping its quarters' value and the trusted declaration: (e) must
  fail.

**Existing assertions.** (S) acts only where a refused leaf's subtree reaches the limit, so most
existing tests should not move. Each changed assertion is listed with the stops that change it. A test
that goes from unconverged to converged, or the reverse, is listed with its honesty at both codes
against the test's own expected answer; it is reported, not a stop.

## 4. Records of the code (Part B)

The log `logs/07c-chase-below-a-refused-leaf.md` in README §5.1's form, with §5's and §6's predictions
(and §9's additions) copied in, marked not yet scored; the board's 07c row; the LiouvilleGreen draws'
harness and outputs in `docs/levin-phase-floor/07c/code/`. **No issue moves in the code commit:** they
move in 07c's records commit after Part C.

## 5. Rehearsal R32 — on the Mac Studio, after the code commit is pushed

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

## 6. The census — on the Mac Studio, Ray-free, on A3 v2's scratch copy

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

## 7. Acceptance

- **Part A:** §2.6's tables complete and its stop reached, with every replay reproducing 07b's census.
- **Part B:** §1 as amended; every test of §3 passing and every breakage caught; the seven roots rising
  only by the tests added; the LiouvilleGreen draws with no new failure.
- **Part C:** §5's S0′–S4 and §6's Y0 hold; Y1–Y9 scored (Y9 added in §9.5), each a verdict against its rule.

## 8. Stop conditions — stop and ask

- A replay at 07b's code that does not reproduce its census record.
- A capture that does not check against 07b's census.
- Part A's projection beyond 36 hours, before the rest runs.
- In Part B: an acceptance test that cannot pass without a change outside §1 as amended; a
  LiouvilleGreen test failing at the commit that passes at the parent; an output of the code differing
  from the chosen variant's replay record on a call the review replays.
- In Part C: S0′ failing; the census's projection beyond 48 hours; a call Y1 says is untouched
  differing from 07b's, or Y8 failing, before anything else is scored; a lock held.

## 9. Amendments at §2's stop

*Written by the orchestrator on 2026-11-19, at §2.7's stop and after Part A's extension on R2-28, from
the user's words, quoted.* Part A's records, with the extension's, are in
`docs/levin-phase-floor/07c/measurement/`. The tables of record are `out/tables/scores_07c.md`,
`out/d1/d1_tables.md`, `out/d2/d2_table.md`, `out/variants/reach.md` and `out/r2-28/summary.md`. What stays on the
Mac Studio, among it every variant's replays but the tree's and (S)'s, is in
`docs/levin-phase-floor/07c/KEPT_ON_MAC_STUDIO.md`.

### 9.1 The rulings

- **At §2.7's stop (2026-11-19):** the orchestrator relayed §2.6's tables. N, Eg and Ea qualify. S, S2,
  SEg and SEa fail on R2-28 alone, converged at 1.12–1.14 of their abserr, while every 07b-lineage code
  is off by the same −2.30×10⁻³⁷. The user: *"Yes, investigate R2-28 first. The S family look to be
  close to the solution we need, and R2-28 doesn't appear to be their fault. So saying the don't
  qualify is a legalistic rather then useful reading of the evidence."* Part A's agent was given the
  extension: where the error sits, why, why 07a's code was right, and whether it is one call or a
  class. §2.6's qualification rule was not changed.
- **After the extension (2026-11-19):** *"I would choose S and open `R2-28` as an issue that is
  othogonal to the fix that S is providing, unless your evaluation is different. We can look at
  `R2-28` later; it's still a relatively small error in the big picture, and the fix is to find a way
  of surfacing the true error"*. The orchestrator's evaluation was the same.

### 9.2 What Part A and its extension measured

**Checks.** Every replay at 07b's code reproduces its census record bit for bit, 454 of 454. The
orchestrator's re-checks (`orchestrator/prompt-07c.md` §A.3) re-ran four replays in `float.hex`,
recounted D1's descent by interval containment, re-drew D2's sample and R4's and R5's selections from
their seeds, re-measured three D2 leaves, re-scored every scored call in exact `Fraction`s from the
readings, and re-ran S and N on two chase calls. All agree with the agent. For the extension, it
re-measured R2-28's carrying leaf against its own Gauss–Legendre reference under the tree and under
S, and recounted the class count from the scores. Both agree.

**The chase (D1).** 192 of the 198 newly unconverged calls end with every depth-20 leaf (668, all
Clenshaw–Curtis) below a leaf a rate rule refused, as 07b's census agent found. Below refused leaves
there are 24,109 bisections, 15,445 Levin and 8,664 Clenshaw–Curtis.
- The Clenshaw–Curtis leaves sit at the noise level: $\delta_1/\nu_B$ has median 0.65–0.96 by cause.
  Most are untrusted by the $r \ge 1$ clause, which is random there. The round-off exit is not
  considered in 6,615 (residual above the floor), and the halving test refuses 2,049.
- The chase is mostly Levin bisection: $\delta_1/\nu_B$ has median 47 on its 15,445 Levin
  bisections (`issues.md` item 4).
- The six calls with no depth-20 leaf miss their budget at 1.000–1.005 of their request.

**The Clenshaw–Curtis estimate (D2).** The fallback's pair is CC₂₄/CC₄₇ (deviation 5). No candidate
estimate bounds CC₄₇'s true error on every sampled leaf. The minimum ratios are: the pair 0.0199,
the third nested rule 0.00408, the tail of the coefficients 0.199 (`tail_sum4`) and 0.118
(`tail_max4x2`). E was run with `tail_sum4`, flagged as having no bound basis.

**The variants** (§2.6; 454 calls; 305 referenced, 299 scored):

| variant | table 3: honest / dishonest / unconverged | chase unconverged of 201 (every depth-20 leaf below a refused leaf) | regions / tree | wall / tree |
|---|---|---|---|---|
| tree | 235 / 0 / 64 | 201 (195) | 1.000 | 1.000 |
| **S** | **292 / 4 / 3** | **5 (0)** | **0.745** | **0.726** |
| S2 | 292 / 4 / 3 | 6 (0) | 0.796 | 0.773 |
| N | 285 / 0 / 14 | 21 (19) | 0.884 | 0.879 |
| Eg, Ea | 228 / 0 / 71 | 186 (182) | 1.313, 1.312 | 1.435, 1.432 |
| SEg, SEa | 287 / 4 / 8 | 4 (0) | 0.926, 0.925 | 1.003, 1.036 |

The four dishonest rows under S, S2, SEg and SEa are R2-28 at both rtols, each scored twice (its
census call `c-*-0-13757-5` and Part A's `a:R2-28`).

**R2-28 (the extension).**
- **Where the error sits.** One leaf carries it: a depth-4 trusted Levin leaf on
  [22.3567, 23.5393] (tree leaf 295, S's 99). Its error is −2.338×10⁻³⁷ against an $e'$ of
  1.874×10⁻³⁷, so 1.25 of its $e'$. Every one of the eight codes makes this leaf, with the same value
  in hex. The rest of each partition errs by +3.6×10⁻³⁹, 0.018 of the call's abserr.
- **Why.** This is 07b's mechanism, an error common to the leaf's halves and quarters. Halves and
  quarters agree, so $r = 0.0048$, partly by cancellation. $e' \approx \delta_1$ is the difference of
  two same-signed errors. $q = 3.26$ is below the range rule's 10. Levin solves at $N \pm 8$ on its
  pieces scatter by about $10^{-37}$, $10^3$–$10^4$ times their floor. So the error is neither
  truncation nor modelled by the floor. Integrand noise and collocation were not separated.
- **Why 07a's code was right.** Its one region's error is −4.82×10⁻³⁸, in one quarter, not
  cancellation. Its $e'$, 8.68×10⁻³⁴, is the root's floor and covers the error 18,000 times over.
- **One call or a class.** Of the 299 scored calls, only R2-28 has every code's error above its
  abserr. Four others share an error across all eight codes at 0.54–0.83 of the abserr. At leaf level
  this was measured on R2-28 only.
- **The stop is not involved:** S does not stop the carrying leaf ($\delta_1/\nu_B = 2.7\times10^4$).

**S on Part A's 454 calls.**
- **Reach.** It touches 328; the other 126 are bit-identical to the tree. Of R2's 60 random calls it
  touches 13. It changes no window's stop or leaves; in 41 windows only the split counter at which the
  stop fell moves.
- **The chase.** None of the 201 chase calls ends with every depth-20 leaf below a refused leaf. The
  five still unconverged are five of the six budget misses, untouched and bit-identical to the tree.
  The sixth, `c-1e-8-0-33997-4`, converges.
- **Honesty.** Its three unconverged referenced calls are `c-1e-8-3-13490-2` (0.0086),
  `c-1e-8-3-13615-3` (0.0083) and v04 (0.000156). R31's record converges at 86 regions (143 at the
  tree), honest at 0.053.

### 9.3 The constants' basis, and where it does not reach

- **`_LEVIN_CHASE_NOISE_FACTOR = 4.0`.** Measured on D1's 8,664 Clenshaw–Curtis bisections below
  refused leaves. Both differences are within $\nu_B$ in 15%, $2\nu_B$ in 73%, $4\nu_B$ in 88% and
  $10\nu_B$ in 97%. $\delta_1/(\phi_P + \phi_L + \phi_R)$ has 90th percentile 1.01, so 4 is about four
  times it.
  - **S2 ($C = 2$)** gives the same table 3, one more chase call unconverged and 7% more regions: the
    verdicts do not turn on $C$ between 2 and 4.
  - **A Levin leaf is rarely at the noise level:** both differences are within $4\nu_B$ in 2,460 of
    15,445 Levin bisections. Otherwise it is refined, not stopped (§0.3's lesson from (i)).
- **`_LEVIN_CHASE_STOP_LIMIT = 6`.** This is dqage's `iroff1` limit, as 07a's window stop counts at the
  same noise level. **It is not measured here:** no variant changed it.
- **Reach.** Every chase call in §2.1 is a group call. The analytic calls have no chase in the sample.
  Where S acts on an analytic call, it is scored only by the referenced analytic calls among Part A's
  (R2's 21, R3's), none of them converged and dishonest under S.
- **The declarations (§1 item 4).**
  - **What was not measured:** a stopped leaf's declaration against a reference, leaf by leaf. It was
    measured at call level only: no referenced call is converged and dishonest under S except R2-28,
    whose carrying leaf S does not stop.
  - **A stopped leaf declares exactly what the tree declares** for the same leaf at `depth_max`. On a
    Clenshaw–Curtis leaf that declaration rests on the pair. D2 shows the pair can understate CC₄₇'s
    true error: in the chase stratum, 5 of 14 resolved leaves have a true error above both the pair
    and the floor. This is `[06d-cc-nested-estimate-measures-the-coarse-rule]`'s, and holds at the
    tree wherever the fallback declares. S adds no declaration the tree does not make, and no
    candidate measured here would bound it.

### 9.4 The census's cost

07b's census took 45,016 s (12.5 h). On Part A's 454 calls S takes 0.745 of the tree's regions and
0.726 of its wall. **§6's harness projects the census** from Y0's K0 run at the new code; beyond 48
hours is a stop, as §6 says.

### 9.5 Additions to §5's and §6's predictions

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

### 9.6 The issues

- **Opened at this stop, on the user's word** (*"open `R2-28` as an issue that is othogonal to the fix
  that S is providing … the fix is to find a way of surfacing the true error"*):
  `[07c-r2-28-trusted-levin-leaf-hides-a-common-error]`, from the agent's `issues.md` items 7, 11 and
  12.
  - A trusted Levin leaf whose rate range is under the range rule's 10 can carry an error common to
    its halves and quarters. Its trust check then sees $r \ll 1$ and declares $e' \approx \delta_1$,
    below the error.
  - On R2-28 one depth-4 leaf is 1.25 of its $e'$, and the call's error is 1.10–1.14 of the abserr
    under every 07b-lineage code. 07a's code was honest only because its single region's floor was
    $10^4$ above the error.
  - Levin solves at low $|\theta'|$ scatter at $10^3$–$10^4$ times their floor, which the floor does not
    model.
  - Next step: a way to surface the leaf's true error. Not assigned.
- `[07b-rate-range-rule-chases-noise-leaves-beyond-the-window-stop]` closes, and
  `[06d-cc-nested-estimate-measures-the-coarse-rule]` is narrowed by §2.3's measurement, in 07c's
  records commit, on the user's word, as the header says.
- **The others** in the agent's `issues.md` (items 1–12) are put to the user at 07c's records commit
  (orchestrator note §C.4).

### 9.7 Stops added (into §8)

- **In Part B:** an output of the code differing from (S)'s replay record on a call the review replays;
  no call found for §3 (b).
- **In Part C:** Y8 failing, before anything else is scored.
