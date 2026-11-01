# Orchestrator — prompt 06g, the trusted leaf's value, the node errors signed, and the premise

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code**, and the deliberate breakages are the agent's, not yours.

**The prompt:** [`prompts/levin-phase-floor/06g-leaf-values-and-node-sum.md`](../06g-leaf-values-and-node-sum.md)
(`e0fc0c6`)
**Board item:** 06g · **Closes, in its records commit, on the user's word:**
- `[06e-trusted-leaf-returns-its-coarsest-value]`, if prompt §2's tests 1–2 and §5's V3 and V4 hold;
- `[06e-node-errors-add-in-absolute-value]`, if §2's tests 4–6 and §5's V5 and V6 hold.

Otherwise ask the user. **Narrows** `[06c-budget-can-stop-on-a-pre-asymptotic-residual]` and
`[07-roundoff-exit-assumes-bisection-cannot-help]`; neither closes. V1, V2, V7 or V8 failing is a
finding for the user, not by itself a reason to keep a header issue open.
**Model:** Opus, for the code agent and for each Part B agent.

This note follows [`prompt-06h.md`](prompt-06h.md), whose machinery it shares: Part A as its §A,
Part B as its §B, with a third Part B agent for the references. It was written 2026-11-01, before
dispatch, at the user's request (*"Can you write the 06g orchestration note, but I will dispatch in
a fresh session."*). **Dispatch only on the user's word**: on 2026-11-01 they said *"Don't dispatch
anything until I say"*, while another 6-core job ran.

## 0. The order

1. **07's work stays set aside:** `stash@{0}`–`stash@{3}` and `var/levin-phase-floor-07/wip/`, as
   06h's note left them. Nothing is set aside again; the four stashes are kept.
2. **Part A** is the code on the laptop, one commit, reviewed here.
3. **Part B**, R23, R24, the census and the references, runs on the Mac Studio after Part A is
   pushed, on the user's word.
4. **07** resumes after 06g's records commit, its rehearsals R25 and R26, its "before" this census.
   The Filon-against-Levin measurement on production's analytic Bessel blocks is a separate prompt,
   not yet written (Decisions, **06g scope**).

## A. Part A — the code, on the laptop

### A.0 Facts, checked by the orchestrator at `e0fc0c6`

- **The code is 06h's** (`7172a8f`, tests `ac1c6f4`); nothing under `AdaptiveLevin/` has changed
  since 06e (`f4d8e17`). The seven roots, from 06h's review (`prompt-06h.md` §A.5) and unchanged
  since: AdaptiveLevin 68, ComputeTargets 780, CosmologyModels 39, LiouvilleGreen 156 (1 skipped),
  ObjectFactories 280, RunDrivers 127, tools 58. AdaptiveLevin (68) and tools (58) re-run here
  at `e0fc0c6`, OK. **Take all seven again at the dispatch SHA,** each on its own.
- **The driver** is `AdaptiveLevin/levin_quadrature.py`'s `_adaptive_levin` (line ~2547). Its
  closures: `_make_leaf`, `_add_leaf`, `_remove_leaf`, `_running_sums`, `_solve_halves`,
  `_check_leaf` (which sums $Q$ as `(LL + LR) + (RL + RR)` and keeps `leaf.value`),
  `_exempt_floor_limited`; the running sums are `_ExactSum`s. The floor-limited test reads
  `region.phase_err` and calls `_roundoff_exit_premise_holds(phase_err, phase_err_L, phase_err_R)`.
- **The floors:** the Levin branch returns `"phase_err": round_off_err + declared_err` (eq. (151)
  plus `_declared_endpoint_phase_err`); the fallback returns the same key as `_cc_roundoff_floor(…,
  abs(E) + R_g + R_2 + R_c) + declared_err`. **$E$ is not returned** today; prompt §1 item 3 adds it.
- **`used_interval.total_err`** is `max(declared_resolution or abserr, phase_err)` plus truncation;
  `_sum_leaves_ascending` sums it.
- **The tests the prompt names** exist: `test_levin_global_budget.py`, `test_levin_roundoff_exit.py`,
  `test_levin_cc_floor.py`, `test_levin_end_nodes.py`, `test_levin_quadrature.py`; 06c's
  `test_converged_flag` Gaussian bump is in `test_levin_quadrature.py` and
  `test_levin_global_budget.py`.
- **The evidence:** sample 8's leaf in `docs/levin-phase-floor/06e/investigations/d3-leaf-values/s08.txt`
  ($\Delta_1 = 2.768\times10^{-17}$, $\Delta_2 = 4.61\times10^{-21}$; with quarters the call's ratio
  is 0.00046, so $Q$'s error is about $1.3\times10^{-20}$, prompt §0.2 (a)); its second leaf,
  $r = 1/3$ on $\Delta_1 = 1.5\times10^{-28}$, is the noise clause's case.
- **06i's tool sources** are `docs/levin-phase-floor/06i/scripts/` (`filon.py`, `route_a.py`,
  `route_b.py`, `r0_closed_forms.py`, `closed_form_a.py`); its readings are `out/readings/`.
- **Z3's 30 calls** are `docs/levin-phase-floor/06c/depth20-tiling/results.json`, `calls` (30
  entries, with `sample_no`, `rtol`, `unit`, `index`).
- **The known flake** is `test_tk_wkb_phase … test_wall_time_per_object`; re-run alone, not a stop.

### A.1 Before you dispatch

1. The user has said to dispatch. `git status --short` is empty; `runregistrykit list` shows
   nothing `running` or stale.
2. Commit this note's addendum, if you write one, and push. That is the dispatch SHA.
3. `touch` a marker in the scratchpad, and record the SHA-256 of the four stashes' commits and of
   `var/levin-phase-floor-07/wip/`.

### A.2 Dispatch

One fresh-context subagent, on **Opus**, in the main checkout. It is given the prompt, the dispatch
SHA, the baselines and §A.0, and it does not read `orchestrator/`. As 06h's note §A.2 (`HEAD` is
not its own; never `git stash`; scope §1–§3, §4's V1 and §5's predictions copied unchanged and not
scored; one commit staged by explicit path, not pushed; no Ray, no store, nothing under `var/`;
every deviation classified; stop at prompt §7), with these paths:
- `AdaptiveLevin/levin_quadrature.py`;
- `AdaptiveLevin/tests/test_levin_leaf_values.py` and `test_levin_node_sum.py`, new;
- `AdaptiveLevin/tests/test_levin_roundoff_exit.py`, additions (test 7);
- `tools/filon_reference.py` and `tools/tests/test_filon_reference.py`, new;
- existing test files under the seven roots, **only** for assertions of the three kinds prompt §2
  test 10 allows, each listed;
- `docs/levin-phase-floor/06g/calibration/`;
- `prompts/levin-phase-floor/logs/06g-leaf-values-and-node-sum.md` and the board's 06g row;
- `docs/OPEN_ISSUES.md` and the board's §3, only if it opens an issue.

Any other path is a stop. It never touches `var/levin-phase-floor-07/`.

### A.3 The review — ten checks, on the laptop

1. **Scope:** `git show --stat` lists only §A.2's paths; the checkout is clean; the four stashes and
   `var/levin-phase-floor-07/wip/` unchanged by SHA-256; nothing under `var/` newer than the marker.
2. **Nothing else moved,** by AST without docstrings: `LiouvilleGreen/`, `ComputeTargets/` and every
   other non-test module untouched; in `levin_quadrature.py`, list every function that changed and
   check each against prompt §1. `tools/filon_reference.py` imports neither `ray` nor a datastore
   package, and nothing outside `tools/` imports it.
3. **The construction, read** against prompt §0.2:
   - **(a)** $Q$ summed in `_check_leaf`'s order; the declared resolution formula unchanged; the
     value sum moved by $Q - V_P$ through `_ExactSum` when a leaf is trusted, and back wherever it
     stops being trusted (floor-limited after checking); $V_P$ kept for untrusted, floor-limited
     and unchecked leaves; `_remove_leaf` subtracts the value the leaf holds;
   - **(b)** the trust test exactly: $r < r_{\rm trust}$, or $r < 1$ and $\Delta_1 \le$ the leaf's
     `phase_err`; $r_{\rm trust} = 1/4$, named, its basis in its comment; the `depth_max`
     declarations for $r < 1$ and $r \ge 1$;
   - **(c)** $E$ added to the signed sum when a fallback leaf is added and removed when it is
     bisected, never twice; $e'_i$ is `total_err` with $|E_i|$ out of the floor only; the budget,
     the ascending confirmation and the returned `abserr` all use $\sum e'_i + |\sum E|$; the
     floor-limited test uses the full floor; the `node_limited` stop is reached only where the
     converged stop's other conditions hold;
   - **(d)** both branches return the declared term; the halving test on $\phi - \phi_{\rm decl}$;
     the budget clause with the halves' floors less $|E|$ and $\sum E$ updated by the halves'; an
     infinite half floor makes the premise hold; 06b's bypass unchanged;
   - **(e)** the tool's routes, implementations, resolutions and count rule as §0.2 (e), and each of
     06i's deviations kept or its change justified.
4. **The roots, re-run here,** one at a time. AdaptiveLevin and tools rise by the new tests; the
   others equal the baselines, or differ only by changed assertions listed under test 10.
5. **The tests, read and run:**
   - tests 1–10 named in the log; **re-run tests 1, 3, 4, 5 and 7 yourself** at the parent and at
     the commit;
   - **on one test-1 cell,** solve one trusted leaf's four quarters yourself with
     `_adaptive_levin_subregion` and check its returned value is their sum in that order, and the
     partition and `abserr` equal the parent's;
   - **on one test-4 cell,** recompute $\sum E$ by `math.fsum` from the returned regions and
     `abserr` as $\sum e'_i + |\sum E|$ yourself;
   - **the Filon tool:** re-check two of 06i's R0 cells ($\omega = 10^6$ and $10^9$) against your
     own antiderivative, as 06i's review did, and one all-orders-½ cell against your own Ci form;
   - read every changed assertion against prompt §2 test 10;
   - re-run the LiouvilleGreen draws in a throwaway worktree and compare with the committed output,
     timings aside.
6. **The breakages:** (i)–(v), each named with its catching test; **re-apply (i), (ii) and (iii)
   yourself** in a throwaway worktree.
7. **Determinism:** one `adaptive_levin_sincos` call with Clenshaw–Curtis leaves and trusted leaves
   (value, abserr, `node_error_signed_sum`, converged and every region's fields), and one Filon
   reading, each twice in fresh processes, bit-identical.
8. **The slow-leaf count:** test 3's descriptive table read; if more than a few per cent of checked
   leaves on the draws sit at $1/4 \le r < 1$ above their floor, report the number to the user with
   the review: it is the cost of prompt §0.2 (b)'s threshold.
9. **The records:** README §5.1's headings, every deviation classified; §4's V1 and §5's Gate V and
   V2–V8 copied verbatim and marked not yet scored; `black` outside `calibration/scripts/`; the
   commit message as `CLAUDE.md`.
10. **Nothing else landed unaccounted:** `git log --oneline <dispatch SHA>..HEAD`.

### A.4 Landing

If the review passes, report to the user. On their word, push the code commit, then commit and push
Part B's dispatch addendum (§B.1), whose commit is R24's and $g$'s tree.

## B. Part B — R23, R24, the census and the references, on the Mac Studio

As 06h's note §B, through an SSH session on the Mac Studio (`ssh macstudio-tunnel`, `--cpus 6`),
with these differences:
- **The scope** is prompt §4 and §5. The worktrees are `SecondaryGWKit-lpf06g-<parent>` and
  `SecondaryGWKit-lpf06g-<dispatch>`.
- **Three agents:** a rehearsal agent (R23, R24, the readers, under the lock), a census agent (Gate V,
  the census, V2 and V5–V8), and a references agent (the capture and Filon references of the 77 new
  samples, the comparison of 06i's 13 with their captures at $g$, V3 and V4). The census and the
  references are Ray-free and need no lock; they do not run at the same time, and each uses at most
  6 processes.
- **The random sample's seed** is chosen and written into the references agent's predictions file
  before the draw; the draw script is copied into the records.
- **Before Part B,** check the Mac Studio's load and the lock: another project's 6-core job ran
  there on 2026-11-01. A held lock is a stop.
- **06h's worktrees** `SecondaryGWKit-lpf06h-2c325f5` and `-c1ec3b2` are no longer needed: R21 and
  R22 are retired, and Gate V's diff needs only commits. Ask the user whether to remove them.
- **A3 v2's scratch copy** is read-only and hashed before and after; nothing in its directory is
  moved, retired or deleted. 06h's census run directory, checkpoint `3104c18e…`, is the "before":
  hash it before and after.
- **The orchestrator re-scores** with its own readers: V1 from R23 and R24; Gate V; V2 and V5–V8
  from the two checkpoints; V3 and V4 from the readings. **One new reference is re-made** another
  way (another resolution, or the other implementation alone at a finer one) and agrees within the
  scatter. On one counted reference with Clenshaw–Curtis leaves, recompute $|v - \sum E - {\rm ref}|$.
- **The census's checkpoint and unit records stay on the Mac Studio** in full; 07 reads them.

### B.1 Part B's dispatch addendum

Written after Part A lands. It records the code commit, its parent, the trees by SHA, and what
Part B must carry.

## 5. Stop and ask the user

As 06h's note §5, with prompt 06g's §7 in place of 06h's. Also stop if the noise clause or the slow
threshold costs convergence on more than a handful of rows (V8), before scoring the rest. **Never
touch 07's stashes or saved work.**
