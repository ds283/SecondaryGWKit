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

## A.5 Addendum, 2026-11-01: the review, the rulings and the landing

**Dispatched** at `4f4f9ed` on the user's word (*"Dispatch prompt 06g … and carry out the review"*),
the seven roots taken there by the orchestrator, each on its own: 68 / 780 / 39 / 156 (1 skipped) /
280 / 127 / 58, all OK, as §A.0. The agent asked no question and stopped nowhere; it committed
**`3ae6f3a`**, not pushed.

**The review of `3ae6f3a`**, the ten checks of §A.3, run by the orchestrator:
1. **Scope:** 19 paths, all §A.2's; the checkout clean; the four stashes and
   `var/levin-phase-floor-07/wip/` (9 files) unchanged by SHA-256; nothing under `var/` newer than
   the marker.
2. **By AST without docstrings:** outside tests only `levin_quadrature.py` and the new
   `tools/filon_reference.py` change. In the module: changed `_ExactSum`, `_adaptive_levin`,
   `_adaptive_levin_subregion_cc`, `_adaptive_levin_subregion_impl`, `_levin_leaf.error`,
   `_roundoff_exit_premise_holds`, `_sum_leaves_ascending` and `used_interval`; added
   `_LEVIN_TRUST_MAX_RATIO`, `_ExactSum.total_with`, `_solve_floor_{declared,roundoff,in_budget}` and
   `used_interval`'s new properties; nothing removed. Each is in prompt §1. The tool imports neither
   Ray nor a datastore (its command line can name `ray.cloudpickle` as the unpickler of a capture,
   the caller's choice); only its tests and `calibration/scripts/breakages.py` import it.
3. **The construction**, read against prompt §0.2: (a) $Q$ as `(LL + LR) + (RL + RR)`, the
   declaration unchanged, the value sum moved through `_ExactSum` by `_set_value` and restored in
   `_exempt_floor_limited`, `_remove_leaf` subtracting the value held; (b) the trust test exactly,
   $r_{\rm trust} = 1/4$ named with the prompt's basis, both `depth_max` declarations; (c) $E$ added
   in `_add_leaf` and removed in `_remove_leaf` only, $e'_i$ = `budget_err` from a floor formed with
   $R_g + R_2 + R_c$ alone (equal to $\phi - |E|$ to $6\times10^{-14}$ relative on test 4's 232
   leaves), $\sum e'_i + |\sum E|$ in the running test, the ascending confirmation and `abserr`,
   the floor-limited test on the full floor, `node_limited` reached only after every leaf is checked
   with none untrusted refinable; (d) both branches return the declared term, the halving test on
   `phase_err_roundoff`, the budget clause in `_split_could_meet_budget`, an infinite half floor
   making the premise hold, 06b's bypass skipping it whole; (e) the tool's routes, implementations,
   constants and resolutions equal to 06i's scripts.
4. **The roots,** each on its own: AdaptiveLevin 82 (+14), tools 63 (+5), the other five equal, all
   OK.
5. **The tests:** 1–7 re-run, 20 tests OK, the parent's numbers as the log's. On test 1's
   `sinc_1e3` cell the orchestrator solved each trusted leaf's four quarters with
   `_adaptive_levin_subregion`: 3 trusted leaves, each returning its quarters sum bit for bit and
   recording $V_P$; the partition, depths, errors and `abserr` equal the parent's; the true error
   $1.095\times10^{-18} \to 5.32\times10^{-19}$ at 40 digits. On test 4's GRZIntegral(1000):
   `math.fsum` of the 232 regions' $E$ ($-5.198\times10^{-18}$; 113 positive, 119 negative) and of
   $|E|$ ($1.297\times10^{-15}$) equal the returned keys, and $\sum e'_i + |\sum E|$ restated from
   the regions equals `abserr` ($4.1724\times10^{-14}$; the parent's $4.3015\times10^{-14}$), both
   unconverged on the same 238 floor-limited leaves. The Filon tool against the orchestrator's own
   antiderivative: R0's cell 1 at $\omega = 10^6$, $W = 10^3$ and $\omega = 10^9$, $W = 10$, all four
   readings within 0.018 of the tolerance; and an all-orders-½ stand-in of its own ($K < 0$,
   $K_{\rm lo} = 2.9\times10^{-12}$, $u \in [0.3, 1.1]$) against
   $A[{\rm Ci}(|K|\eta_b) - {\rm Ci}(|K|\eta_a)]$ and its sine counterpart in Si, within 0.011.
   The two changed assertions are counts that (b) moves. The 20 draws re-run in a throwaway
   worktree: 160 units identical to the committed output but for timings and the module's path; the
   committed run's module hash `d89c959e…` is the committed file's.
6. **The breakages:** (i) re-applied by hand, caught by test 1 on all seven cells and by three more
   tests; (ii) by tests 4 and 5; (iii) by test 7's first case; (iv) and (v) from the record.
7. **Determinism:** `adaptive_levin_sincos` on SincIntegral at rtol $10^{-10}$, order 12 (17
   regions, 8 Clenshaw–Curtis, 16 returning $Q$: value, `abserr`, the node keys, converged and every
   region's fields) and one FCC reading of the Ci case, each bit-identical in two fresh processes.
8. **Slow leaves:** at the parent 670 of the draws' 60,241 checked leaves (1.1 %) sat at
   $1/4 \le r < 1$ above their floor, under the few per cent this check reports; under 06g regions
   rise 61,126 → 73,464 and solves 431,180 → 504,212.
9. **The records:** README §5.1's headings, 16 deviations classified; V1, Gate V and V2–V8
   verbatim, not yet scored; `black` clean; the commit message as `CLAUDE.md`. One slip, in
   `docs/OPEN_ISSUES.md`'s campaign sentence (a lost comma), corrected in this addendum's commit.
10. `4f4f9ed..HEAD` held only `3ae6f3a`.

**The rulings** (the board's Decisions, **06g Part A lands**). Deviation 1, the budget clause held
when the call already meets its request: *"Deviation 1: accept"*. Deviation 6, test 7's first cell:
*"Deviation 6: accept"*. The issue the agent opened,
`[06g-node-limited-stop-can-end-a-call-refinement-would-converge]`: *"run Part B as written and see
what happens"*. **Landed:** `3ae6f3a` pushed. 06h's two Mac Studio worktrees were removed on the
user's word, both clean (§B).

### B.2 Addendum, 2026-11-01: Part B's dispatch (§B.1)

**Part A** is the code commit **`3ae6f3a`**, whose parent is **`4f4f9ed`**, reviewed (§A.5) and
pushed.

**The trees, by SHA:**
- **R23 and Gate V's reproduction** at **`4f4f9ed`**, in the parent worktree. Its code is 06h's
  census tree's: `git diff --stat c1ec3b2 4f4f9ed -- . ':!docs' ':!prompts'` shows only
  `CLAUDE.md` (checked on the laptop).
- **R24, $g$ and the references** at **this addendum's commit**, in the code worktree. Its code is
  `3ae6f3a`'s: the commit adds only records. Gate V's second clause, `git diff --stat c1ec3b2
  3ae6f3a -- . ':!docs' ':!prompts'`, touches `AdaptiveLevin/` (the module and four test files),
  `tools/` (the tool and its test) and `CLAUDE.md`, and nothing else (checked on the laptop).
- **06h's census** `levin-phase-floor-06h-census-20261101T005037` is the "before", its
  `checkpoint.jsonl` `3104c18e…` (`docs/levin-phase-floor/06h/KEPT_ON_MAC_STUDIO.md`).
- **Z3's 30 calls** are `docs/levin-phase-floor/06c/depth20-tiling/results.json`'s `calls`; 06i's
  13 references are `docs/levin-phase-floor/06i/out/readings/`.

**What Part B must carry:**
- **The census harness** wraps `_roundoff_exit_premise_holds` to count premise checks (06h's); the
  function now takes `split_could_meet_budget=`, which a wrapper must pass through (the log's
  Observation 6). Each harness change is a deviation, its diff saved.
- **The tool's command line:** `python -m tools.filon_reference PICKLE --tolerance T --unpickler
  ray.cloudpickle --workers 6`, with $g$ and the census harness on `PYTHONPATH` (the captures name
  the harness's functions). Route B's default edges walk the captured integrand's closures as 06i
  did; a capture whose closures differ raises, and its edges must be passed. The tool was not run
  on 06i's 13 captures on the laptop (the log's deviation 12): §5's comparison with 06i's readings
  is that check, and a disagreement is a §7 stop.
- **V8 may fail.** The issue `[06g-node-limited-stop-can-end-a-call-refinement-would-converge]`
  shows synthetic fallback calls far from the origin that the parent converges and 06g stops
  node-limited; on the laptop's draws two calls lost convergence to (b), none to `node_limited`.
  The user ruled to run Part B as written: each lost call is reported with its cause, and the §5
  stop "the noise clause or the slow threshold costs convergence on more than a handful of rows"
  (§5 of this note) stands.
- **The near-singular class** moved 6 honest → dishonest and 10 the other way on the draws; if A3 v2
  reaches it, report it as such.

**§B.0 re-checked on the Mac Studio at 10:55 UTC, 2026-11-01,** read-only before the removal:
- the main checkout is `handover-remedial` at `28ebaf0`, `git status --short` empty;
- 06h's worktrees `SecondaryGWKit-lpf06h-2c325f5` and `-c1ec3b2` were clean, with no `var/`, and
  were removed on the user's word with `git worktree remove`; `git worktree list` now shows the main
  checkout alone;
- the registry holds 40 runs, all finished, nothing `running` or stale;
- the lock is free (`~/.raypoolkit/` empty), and `ray-processes --ray-temp-dir /tmp/ray-sgk` lists
  none; no other project's job runs (load about 4, background services);
- there is no `var/levin-phase-floor-06g*` and no `var/datastores/levin-phase-floor-06g*`;
- the scratch copy's five files match 06e's certified SHA-256s
  (`06e/review/checks/a3v2_copy_after.sha256`), with no `-wal` or `-shm`;
- 06h's census checkpoint is `3104c18e…`;
- 95 GiB free.

**The worktrees,** made by the laptop after this commit is pushed, detached, with no venv:
`SecondaryGWKit-lpf06g-4f4f9ed` and `SecondaryGWKit-lpf06g-<this commit's short SHA>`.

### B.3 Addendum, 2026-11-01: V1's twelve calls referenced, scored apart from V3

R23 and R24 agree (`1b332ffa…`), and Gate V held. V1 fails on `total` in 12 of the 864 rows, at
worst by 13,101×. In each of those rows the whole move is in a single group call (sub-interval 1,
group `G+q+r`): the value shifts 0.1–3.4% while both codes declare an error near 1e-7. V1 holds on
the analytic oracle, where the worst ratio is 0.4986. The user ruled, in these words: "Happy to
have the Filon references added – that sounds like a good idea." The ruling went to the Part B
session as follows:

- **The calls.** The 12 calls are referenced by Route B under the unchanged count rule. A call
  that cannot be counted is recorded with its four readings.
- **Scoring.** They are scored apart from V3, in their own table: not part of V3's sample, and
  they do not move its verdict.
- **What each row records.** Each counted call records the reference, both codes' values and
  declared errors, and |value − ref| / declared error for each code.
- **The question they answer.** Which code is nearer the truth, and does that code's declared
  error cover its distance from the reference?
- **Stop rule.** If adding them would change how V3's own sample is drawn or scored, the session
  stops and asks.

### B.4 Addendum, 2026-11-01: the V8 stop, and the user's ruling to score the rest

The census `levin-phase-floor-06g-census-20261101T114315` ran clean. All 18,560 units finished
with exit 0, none in error and none over budget, in about 5 h 40 min against 06h's 2 h 33 min.

V8 fails at both rtols. Outside the 720 stationary-point rows, convergence is lost to the
slow-leaf rule (b) as follows:

- at 1e-7, 801 calls in 470 rows;
- at 1e-8, 827 calls in 474 rows, plus 3 calls lost to `node_limited`.

Every lost call is a `G±q±r` group call. Every one still meets its request on abserr: the ratio
of abserr to request has median 0.035 and maximum 0.987. Median regions per call rose from 7 to
135. The Part B session stopped under §5 of this note (its `review/stop_v8.md`).

The user ruled "(b) with the V8 sample, please". The ruling went to the Part B session as follows:

- The stop and V8's failure stand in the record.
- V2, V5–V7, the descriptive items and the references (V3, V4, and §B.3's V1 calls) are scored
  for the record.
- Twelve calls that V8 lost to (b) are also referenced:
  - six at each rtol;
  - all outside the stationary-point rows;
  - chosen at evenly spaced quantiles of abserr/request;
  - with the selection rule and the identities written down before any reference is computed.
- These twelve are scored apart from V3, under Route B and the unchanged count rule. Each row
  records:
  - the reference;
  - R23's value, declared error and converged flag;
  - $g$'s value and declared error;
  - each code's |value − ref| / declared error.
- The question the sample answers: does the value that $g$ returns from untrusted slow leaves lie
  within its declared error of the reference, and does R23's?
- The session stops and asks if either added sample would change V3's own sample, or if the
  references would cost more than the census did.

### B.5 Addendum, 2026-11-01: a descriptive breakdown of the slow leaves

The parent trusted a checked leaf at any $r < 1$ (06c's trust check, `4f4f9ed`). The value 1/2
appears only in 06g's prompt, as the less conservative alternative to 1/4. The leaves that V8
loses were therefore trusted anywhere in $1/4 \le r < 1$, and `v8.md` does not record their $r$.
They were refined to depth 20 and still show $r \ge 1/4$, so refinement never brought them into
the asymptotic regime that the basis for $r_{\rm trust}$ assumes. Two explanations remain open:

- **(i)** a non-smooth point in the integrand or at a boundary, which gives algebraic convergence
  at a steady ratio;
- **(ii)** rounding noise above a $\phi_P$ that is too low.

The user asked for a descriptive item to tell these apart. It was sent to the Part B session. It
covers every returned untrusted (b) leaf of a lost call, outside the 720 rows, at both rtols, and
is computed from the census records with nothing rerun:

- $r$ binned into [1/4, 1/2), [1/2, 3/4) and [3/4, 1), with quantiles;
- $\Delta_1/\phi_P$, overall and in each bin;
- $r$ against depth, along each leaf's chain of ancestors where the records hold it, and across
  leaves where they do not;
- the depth-20 leaves' positions: clustered at a few points per call, or spread, and whether the
  clusters sit at sub-interval or region endpoints;
- everything split by group.

Anything the records cannot give is reported as such, not rerun.

### B.6 Addendum, 2026-11-01: the references' part 2, re-planned by the user

**Part 1** passed every check:
- all 102 captures equal the census bit for bit;
- 06i's 13 groups equal 06i's;
- the tool agrees with 06i's readings on all 13.

**Part 2** stopped at the cost gate at 20.35 h. Of that, 15.5 h was 31 QAWO readings on Route A
calls with phase spans up to $1.6\times10^{12}$ rad. They were predicted to pass §5's 30-minute
cap, which would have killed them with no result. QAWO's panels are uniform in phase, so its time
grows with the span. FCC's cost does not grow with the span.

**06i's 13**, where FCC, QAWO and Levin all exist, give this comparison:
- FCC and QAWO agree to between $10^{-17}$ and $10^{-13}$ relative, and match the closed forms on
  s08, s10 and s22 to about $10^{-16}$;
- 06e's Levin values sit $5\times10^{-14}$ to $1.2\times10^{-6}$ from them, at or below their
  declared errors except s03, at 1.24×.

**The user ruled:**
- "Quibbling over ~8 hours of compute that only has to be run once strikes me as a false economy";
- "A runtime that exceeds 30 minutes means that we should stop and check, not guillotine it
  immediately";
- "We are iteratively improving on our algorithm, and we don't know what to expect."

**As sent to the Part B session:**
- **Large-span calls.** QAWO r2 is not run on the 17 large-span calls. QAWO r1 is run where its
  projection is under about 2 h: z20, r30, z19, z26, z05, z17, z11, z16, z28 and r07. FCC is run at
  both resolutions on all 17. The projection is about 13 h of wall time, with the cheap readings
  first.
- **The 30-minute cap.** It becomes a check. A reading past it is reported with its progress and
  projected finish. It finishes if that projection is reasonable, and is stopped only if the
  projection is absurd, with a backstop recorded. One check is made at the projected end.
- **Scoring.** Every reference is scored, whatever its number of readings, and its readings are
  stated. Most weight goes to references with two independent implementations. An FCC-only
  reference far from Levin's value, or with its two resolutions disagreeing, is a finding to stop
  on, never a reason to drop the reference.

### B.7 Addendum, 2026-11-03: the end of Part B, and the records

**The rerun batch** `levin-phase-floor-06g-references-rerun-20261102T185324` ended at 06:47Z on
3 November, exit 0: 64 readings done and 3 stopped. Stage 2's 56 stationary-point readings ran at
`ff3e8394`.

**The laptop's FCC batch** `levin-phase-floor-06g-fcc-laptop-20261102T233229` (32 of 32, at
`b5626ecb`) re-read FCC on the 14 stationary-point samples, after four bit-for-bit checks against
the Mac Studio's readings. The four readings of each of the 14 now agree.

**The rescoring:**
- V1's 12: R24 is honest on all 12, and R23 is $4.5\times10^3$–$6.0\times10^5$× out.
- V3 holds as written, and fails on z12 alone under the broader readings. z12 is a stationary-point
  row, converged at 1,681× its error.
- V4 holds.

**Stage 3.** r24's and r53's QAWO r1 readings landed. r47 r2, r53 r2 and r26 r1 were stopped by a
projection that ignored the chunks in flight. None of them bears on a verdict, and their rerun is
for the user.

**The Mac Studio review** (`06g/review.md`) passed its ten checks.

**The user's reading:** z12 and V1's 12 are stationary-point calls, where Levin's declared errors
are not claimed valid. *"What needs to be true is that z12 works after 07."*

**On the user's word, the records were committed.** They were copied back and checked by SHA-256
(2,811 files; `review/checks/var_files_after.txt` brought over gzipped). The two issues the review
proposed for V5, V6, V8 and V2 are opened, six issues are narrowed, and none is closed. The 71
stray `.pyc` files of the laptop's slope survey in the `ff3e839` worktree, and the one in
`references/scripts/__pycache__/`, were deleted.
