# Orchestrator — prompt 06m, the signed trust yardstick in the driver

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code**, and the deliberate breakages are the agent's, not yours.

**The prompt:** [`prompts/levin-phase-floor/06m-signed-yardstick-in-the-driver.md`](../06m-signed-yardstick-in-the-driver.md)
**Board item:** 06m · **Closes, in its records commit, on the user's word:**
`[06g-slow-leaves-at-half-cost-convergence-outside-stationary-points]`, if prompt §2's tests 1–5,
§5's Gate M and §4's R1 hold. At that commit propose its residual,
`[06m-calls-flagged-unconverged-within-their-request]` (prompt §0.4), and open it only on the
user's word. R1 failing is a finding for the user, not by itself a reason to keep the issue open.
**Model:** Opus, for the code agent and for each Part B agent.

This note follows [`prompt-06g.md`](prompt-06g.md), whose machinery it shares: Part A as its §A,
Part B as its §B, with two Part B agents (no references are re-made: Gate M certifies the code by
identity with 06l's census). Part B runs through an orchestrator session on the Mac Studio, as 06k's
and 06l's did. **Dispatch only on the user's word.**

## 0. The order

1. **07's work stays set aside:** `stash@{0}`–`stash@{3}` and `var/levin-phase-floor-07/wip/`. They
   are kept, and nothing touches them.
2. **Part A** is the code on the laptop, one commit, reviewed here.
3. **Part B** (R25, R26 and the census) runs on the Mac Studio after Part A is pushed, on the user's
   word.
4. **07** resumes after 06m's records commit. Its rehearsals become R27 and R28, and its "before" is
   this census.

## A. Part A — the code, on the laptop

### A.0 Facts, checked by the orchestrator at `7d7f7af6`, 2026-11-06

- **The code is 06g's.** `git log 3ae6f3a4..HEAD -- AdaptiveLevin/` is empty, so 06l's variant
  differs from the tree only by its diff, `docs/levin-phase-floor/06l/out/variant_routeB_diff.txt`.
  Since 06g, `tools/` changed (`ff3e8394`, `b5626ecb`). **Take all seven roots again at the dispatch
  SHA,** each on its own.
- **The trust check** is `_check_leaf` in `_adaptive_levin` (line ~3322). It reads
  `region.phase_err` ($\phi_P$) and the halves' dicts `dataL` and `dataR`. `leaf.is_direct` is a
  slot of `_levin_leaf`. The halves' `is_direct`, `phase_err_less_node` and `node_error_signed`
  come from `_adaptive_levin_subregion_cc`'s return dict (line ~2684).
- **The variant** is `docs/levin-phase-floor/06l/scripts/variants/levin_quadrature_routeB.py`,
  SHA-256 `5aa6178a2298fb4cc32620cf6ae81538e873bf6a3177e8db6f3a5274d9bb4d9f`. Its trust test is
  hunk 2 of the diff. Its `route_records_06l` and `route_06l` are instruments, not ported.
- **06l's LiouvilleGreen harness** is `docs/levin-phase-floor/06l/scripts/calibration/`
  (`lg_seeded_draws.py` with modes for the tree and a variant, `lg_compare_06l.py`, `prov.py`).
  Under B, 122 of 4,320 Levin calls fire, with no convergence or honesty change.
- **The driver's result dict** has one timing field, `elapsed`.
- **The known flake** is `test_tk_wkb_phase … test_wall_time_per_object`; re-run it alone, not a
  stop.

### A.1 Before you dispatch

1. The user has said to dispatch. `git status --short` is empty, and `runregistrykit list` shows
   nothing `running` or stale.
2. Commit this note's addendum, if you write one, and push. That is the dispatch SHA.
3. `touch` a marker in the scratchpad, and record the SHA-256 of the four stashes' commits and of
   `var/levin-phase-floor-07/wip/`.

### A.2 Dispatch

One fresh-context subagent, on **Opus**, in the main checkout. It is given the prompt, the dispatch
SHA, the baselines and §A.0, and it does not read `orchestrator/`. As 06g's note §A.2: `HEAD` is not
its own; it never runs `git stash`; its scope is §1–§3, with §4's R1 and §5's Gates copied unchanged
and not scored; it makes one commit, staged by explicit path, not pushed; no Ray, no store, nothing
under `var/`; every deviation is classified; and it stops at prompt §7. Its paths are:
- `AdaptiveLevin/levin_quadrature.py`;
- `AdaptiveLevin/tests/test_levin_trust_yardstick.py`, new;
- existing test files under the seven roots, **only** for assertions of prompt §2 test 7's kind,
  each listed;
- `docs/levin-phase-floor/06m/calibration/`;
- `prompts/levin-phase-floor/logs/06m-signed-yardstick-in-the-driver.md` and the board's 06m row;
- `docs/OPEN_ISSUES.md` and the board's §3, only if it opens an issue.

Any other path is a stop. It never touches `var/levin-phase-floor-07/`.

### A.3 The review — nine checks, on the laptop

1. **Scope:**
   - `git show --stat` lists only §A.2's paths, and the checkout is clean;
   - the four stashes and `var/levin-phase-floor-07/wip/` are unchanged by SHA-256;
   - nothing under `var/` is newer than the marker.
2. **Nothing else moved,** by AST without docstrings:
   - `LiouvilleGreen/`, `ComputeTargets/` and every other non-test module are untouched;
   - in `levin_quadrature.py`, the only changes are the new `_trust_noise_yardstick` and
     `_check_leaf`'s trust statement.
3. **The construction, read** against prompt §0.1:
   - the four cases;
   - a Clenshaw–Curtis half adds nothing on a Levin parent;
   - $\phi^-$ is `phase_err_less_node`;
   - the signed sum is over the Clenshaw–Curtis halves only;
   - the order of the additions;
   - a `None` Levin $\phi_h$ is read as 0.0;
   - the raise.

   **Then your own check:** put `_trust_noise_yardstick` beside the variant's hunk 2, line by line.
4. **The roots, re-run here,** one at a time. AdaptiveLevin rises by the new tests; the others equal
   the baselines, or differ only by changed assertions listed under test 7.
5. **The tests, read and run:**
   - tests 1–7 are named in the log; **re-run tests 1–5 yourself** at the commit;
   - **test 2, independently:** on two cells of your own choice, one with a Clenshaw–Curtis-parent
     firing and one with a Levin-parent firing, run the commit and the variant yourself and compare
     every result key (`elapsed` aside) and every region field bit for bit;
   - read every changed assertion against test 7;
   - re-run the LiouvilleGreen draws in a throwaway worktree in the commit's mode and the variant's,
     and check they are equal.
6. **The breakages:** (i)–(v), each named with its catching test. **Re-apply (i), (ii) and (iii)
   yourself** in a throwaway worktree.
7. **Determinism:** one `adaptive_levin_sincos` call with a firing, twice in fresh processes,
   bit-identical (`elapsed` aside).
8. **The records:**
   - README §5.1's headings, with every deviation classified;
   - §4's R1 and §5's Gates M0 and M copied verbatim and marked not yet scored;
   - `black` outside `calibration/scripts/`;
   - the commit message as `CLAUDE.md`.
9. **Nothing else landed unaccounted:** `git log --oneline <dispatch SHA>..HEAD`.

### A.4 Landing

If the review passes, report to the user. On their word, push the code commit, then commit and push
Part B's dispatch addendum (§B.1), whose commit is R26's and $m$'s tree.

## B. Part B — R25, R26 and the census, on the Mac Studio

Through an orchestrator session on the Mac Studio, started by the laptop in the code worktree, as
06l's was. Its folder must be trusted in the app first, and it reports to the laptop session.
Differences from 06g's note §B:
- **The scope** is prompt §4 and §5. The worktrees are `SecondaryGWKit-lpf06m-<parent>` (R25) and
  `SecondaryGWKit-lpf06m-<dispatch>` (R26 and $m$), made by the laptop, detached, with no venv; the
  main checkout's venv is used.
- **Two agents, one at a time:**
  - a rehearsal agent: R25, R26, R25 against R24, the readers, under the lock;
  - then a census agent: Gate M0, the census, Gate M and the descriptive tables, Ray-free, with
    no lock, at most 6 processes.
- **Planned stops,** each checked by the Mac Studio orchestrator before the agent goes on:
  1. after R25 and R26 and the value comparisons, before the readers and the lock's release:
     - both fingerprints;
     - R25 against R24;
     - outside `QuadSourceIntegral`, identical value tables;
     - R1 re-scored by the orchestrator's own reader;
  2. after Gate M0, before the census: re-run two of the 100 K0 units yourself and compare them
     with route B's census;
  3. after the census: `done`, 18,560 of 18,560, before Gate M is scored.
- **Wait on every detached run** with a background `kill -0` loop on its pid, never by polling.
- **Before Part B,** check the Mac Studio's load and the lock. A held lock is a stop.
- **A3 v2's scratch copy** is read-only and hashed before and after; nothing in its directory is
  moved, retired or deleted. Route B's census run directory (checkpoint `5cb3d3c0…`) is the
  "before": hash it before and after. 06g's census (`03adcbba…`) is read for the descriptive
  tables, and hashed the same way.
- **The orchestrator re-scores** with its own readers: R1 from R25 and R26; Gate M from the two
  checkpoints, by its own comparison of every call record and region record; the flag-cost table
  from the census checkpoint.
- **The census's checkpoint and unit records stay on the Mac Studio** in full; 07 reads them.
- **Never touch** 06g's four worktrees, `var/levin-phase-floor-07*`, R23 and R24, or 06g's, 06k's
  and 06l's records and runs, which are read only.

### B.1 Part B's dispatch addendum

Written after Part A lands. It records the code commit, its parent, the trees by SHA, and what
Part B must carry.

## 5. Stop and ask the user

As 06g's note §5, with prompt 06m's §7 in place of 06g's. Also stop if a single call of the census
differs from route B's, before scoring anything else. **Never touch 07's stashes or saved work.**
