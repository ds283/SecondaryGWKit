# Orchestrator — prompt 06d, a round-off floor for the Clenshaw–Curtis rule

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code**, and the deliberate breakages are the agent's, not yours.

**The prompt:** [`prompts/levin-phase-floor/06d-cc-roundoff-floor.md`](../06d-cc-roundoff-floor.md)
**Board item:** 06d · **Closes, in 06c's records commit, if 06c's §4 and X0–X4 hold:**
`[07-cc-solves-use-the-levin-roundoff-floor]` and `[07-depth18-diagnostic-divides-by-zero]` ·
**Model:** Opus.

This note follows [`prompt-06c.md`](prompt-06c.md) §A, whose machinery it shares. **06d has a
Part A only**: its rehearsals and census are 06c's Part B, run after it (06c's note §B.6).

## 0. The order

1. **07's work is set aside a third time.** Its §1, rebased onto 06c by its agent at its third stop,
   is uncommitted in the main checkout: the module staged as resolved, the other two files staged,
   `docs/levin-phase-floor/07/` and its log untracked. Before this dispatch the laptop saves it as a
   patch and a tarball in `var/levin-phase-floor-07/wip/`, recorded in `SHA256SUMS.txt`, and as a
   stash naming 07, 06d and its base. The earlier two stashes are kept.
2. **Part A** is the code on the laptop, one commit, reviewed here.
3. **06c's Part B** follows, on the commit that records 06d's landing (06c's note §B.6).
4. **07 resumes** on 06d's code, with its §2 re-run (07 §9 point 7).

## A. Part A

### A.0 Facts, checked by the orchestrator at the dispatch SHA

- **The code is 06c's** (`ad9a5f4`); the seven roots are AdaptiveLevin 53, ComputeTargets 780,
  CosmologyModels 39, LiouvilleGreen 148 (1 skipped), ObjectFactories 280, RunDrivers 127, tools 58.
- **The fallback's floor** is formed in `_adaptive_levin_subregion_cc` from `_levin_G0_G1` and
  `_roundoff_floor` on the coarse `theta_prime_Cheb`, plus `_declared_endpoint_phase_err` with
  $\tfrac12\|f\|h$ per endpoint. `eval_basis` hands sin and cos `theta_mod_2pi(x)` when supplied,
  else `theta(x)`.
- **The depth-18 log** divides at `abs_improvement = prev_abs / this_abs` in `_make_leaf`.
- **The known flake** is `test_tk_wkb_phase … test_wall_time_per_object`; re-run alone, not a stop.

### A.1 Before you dispatch

1. Set 07's work aside (§0 item 1), and check `git status --short` is empty.
2. Commit the prompt, this note, the amendments and the board, and push. That is the dispatch SHA.
3. `touch` a marker in the scratchpad.

### A.2 Dispatch

One fresh-context subagent, on **Opus**, in the main checkout, given the prompt, the dispatch SHA,
the baselines and §A.0; it does not read `orchestrator/`. As 06c's §A.2, with these paths:
- `AdaptiveLevin/levin_quadrature.py`;
- `AdaptiveLevin/tests/test_levin_cc_floor.py`, new;
- existing test files under `AdaptiveLevin/tests/` and `LiouvilleGreen/tests/`, only as prompt §2
  test 8 allows, each listed;
- `docs/levin-phase-floor/06d/calibration/`;
- `prompts/levin-phase-floor/logs/06d-cc-roundoff-floor.md` and the board's 06d row;
- `docs/OPEN_ISSUES.md`, only if it opens an issue.

Any other path is a stop. It never runs `git stash`, never touches `var/levin-phase-floor-07/`,
uses no Ray and no store, writes nothing under `var/`, and makes one commit, not pushed.

### A.3 The review — on the laptop

1. **Scope:** `git show --stat` lists only §A.2's paths; the checkout clean; the three stashes and
   `var/levin-phase-floor-07/wip/` as left; nothing under `var/` newer than the marker.
2. **Nothing else moved,** by AST without docstrings: only `_adaptive_levin_subregion_cc`, the
   depth-18 log in `_adaptive_levin`, and any new private helper for the floor; the result keys
   unchanged.
3. **The construction, read** against prompt §0.2: the two terms, $n = 2N-1$, $(n + 4)$ stated term
   by term, $u = 2^{-53}$, the arguments those `eval_basis` uses, no $1/\theta'$, the declared term
   unchanged; the guard logs only.
4. **The roots, re-run here.** Six equal the baselines unless an allowed assertion changed;
   AdaptiveLevin rises by the new module's methods.
5. **The tests, read and run:** tests 1–8 named in the log; **re-run tests 1 and 4 yourself at the
   parent and at the commit**; read every changed assertion against prompt §2 test 8; re-run the
   LiouvilleGreen draws in a throwaway worktree and compare with the committed output, timings
   aside.
6. **The breakages:** (i)–(iii) each named with their catching test; **re-apply (i) yourself.**
7. **Determinism:** test 4's problem twice in fresh processes, bit-identical.
8. **The records:** the log's README §5.1 headings, every deviation classified; `black`; the commit
   message as `CLAUDE.md`.
9. **Nothing else landed unaccounted:** `git log --oneline <dispatch SHA>..HEAD`.

### A.4 Landing

If the review passes, push the code commit, then one residue commit with its hash on the board and
in the log and 06c's §B.5 trees revised for Part B (06c's note §B.6). Then 06c's Part B is
dispatched on the user's word, and 07's work is restored for its agent from the newest stash.

## 5. Stop and ask the user

As 06c's note §5, with prompt 06d's §4 in place of 06c's §7. **Never touch 07's stashes or saved
work** except to restore the newest stash as §A.4 says.

## A.5 Addendum, 2026-10-29: the stop and the amendment

Part A stopped under prompt §4 with nothing committed: $\phi_{\rm CC}$ missed the rounding of the
nodes, and two existing tests rested on premises 06d removes. The user agreed the three remedies
(*"I agree with all three"*), written as prompt §5; the same agent resumes on its uncommitted work.
The review adds to §A.3:
- **Check 3** reads the node term: $\delta_k$ counted against how the code forms the nodes, both
  parts of $g'$ present, no $1/\theta'$.
- **Check 5** reads the three-Bessel restatement and 06b's test 3 against prompt §5 points 2 and 3,
  and re-runs test 2's reduced-phase cells myself, with the node term and with it removed.
- **Check 6** names breakage (iv) with its catching test.
