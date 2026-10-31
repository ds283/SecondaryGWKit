## Your brief: the census agent — prompt 06e §5 (06c's census re-run under $g$, and Z3's references)

**Your scope is prompt 06e §5**, and only it. §4 (rehearsals R19 and R20, under Ray and the lock)
is the other agent's, and runs first.

- **Where you write:** only `var/levin-phase-floor-06e/census/` (with `out/`, `scripts/`, `state/`,
  and `references/`) and your own run directories in `var/runs/`. You take no lock, and no process
  of yours initialises Ray (importing the `ray` package, as production modules do, is fine; a Ray
  session appearing is not).
- **The harness** is 06c's census copy, from `docs/levin-phase-floor/06c/census/scripts/` (in the
  code worktree), copied into `var/levin-phase-floor-06e/census/scripts/`. Change only what prompt
  §5 allows (the tree and its paths; what Z2–Z5 need recorded per call). The diff against 06c's copy
  is saved in `out/`, and each change is a numbered deviation.
- **The trees:** Z0's reproduction of 06c runs at the parent worktree (`50277f3`); $g$ is the code
  worktree (`6d81fad`). 06c's census tree was `e5764a6`.
- **The "before"** is 06c's census run `var/runs/levin-phase-floor-06c-census-20261029T130214/`,
  read from its `checkpoint.jsonl` (`b61cb034…`) and unit pickles. Read it only, and confirm its
  checkpoint's SHA-256 before and after.
- **The store** is the scratch copy, opened read-only, re-checked against its certification and
  fingerprint before the run (with 06c's copied `fingerprint_copy.py` or its equivalent), and hashed
  (SHA-256) before and after, with no journal, `-wal` or `-shm` file beside it.
- **Gate Z0 runs before the launch,** all three clauses of prompt §5:
  - the first is against `e5764a6`; the list of modules the harness imports is taken from a dry run
    at the parent;
  - the second (06c's units `low-G+r-1` and `ctrl-low-1` at `1e-7`, `0-1971` at `1e-8`, reproduced
    at the parent) and the third (two runs of one $g$ unit) are compared by bit pattern
    (`float.hex`).

  Save Z0's outputs in `out/`.
- **The predictions** go in `state/predictions.md`, **after Z0 and before the launch**: prompt §5's
  Z2–Z5 text, unchanged (copied, not paraphrased), with "the stationary-point rows" fixed as 06a's
  720 (`spscan.py`) and Z3's sample list and matching rule (row, rtol, call index) stated
  mechanically. Its mtime is the review's evidence; do not edit it after the launch. Anything you
  add after the launch goes in another file.
- **What each record carries.** Every checkpoint record carries its script's SHA-256, its tree's
  `HEAD` and `git status --short`, and each unit's evidence of where its modules were loaded from
  (so a reader can see a $g$ unit imported from the code worktree and a parent unit from the parent
  worktree). Per call, record what Z2–Z5 and the descriptive items need — each returned leaf's depth,
  branch (Levin or Clenshaw–Curtis), residual, floor (and for Clenshaw–Curtis leaves each of the
  floor's terms, as 06e's code exposes them), truncation, `total_err`, and the call's value, abserr,
  `converged`, regions and solves — so that the orchestrator can re-score Z2, Z4 and Z5 from the two
  checkpoints with its own reader. State in your log where each field lives.
- **At most six worker processes.** A Ray session under `/tmp/ray-sgk` (or anywhere) started by a
  census process is a stop.
- **The budget** is 300 s per unit, enforced from the dispatcher, with 06a's over-budget pass, as
  06c ran it. Over-budget units count against Z2 as the prompt says; do not change the budget.
- **The run** is one registered, detached run with slug `census`
  (`--campaign levin-phase-floor --prompt 06e`), job name `levin-phase-floor-06e`, as 06c's was
  launched.
- **The checkpoint and per-unit records (pickles) are kept in full** in your run directory and
  records. 07's §6 reads them later. Do not prune them.
- **Z3's references** (prompt §5, Z3): `orch_ref.py`'s method, copied into
  `var/levin-phase-floor-06e/census/references/` (record its SHA-256 against 06c's and any diff as a
  deviation), on `depth20-tiling`'s captured inputs, at three resolutions.
  - Samples 1, 9 and 25 are **re-scored** against $g$ from the existing references, not re-run.
  - Samples 2, 8, 10, 13, 14, 22, 23 and 27 run **in-session** (unregistered), **one at a time**,
    each detached (`nohup`), its output in `references/`. Launch one, make one check, and end your
    turn with its pid; you are resumed when it exits. (Several of them in one detached sequential
    script is allowed if you record it as a deviation; never two at once.)
  - Samples 3 and 15 run in **one registered, detached run** with slug `references`
    (`--campaign levin-phase-floor --prompt 06e`), after the short ones.
  - Every reference uses at most six workers, and none runs while the census or another reference
    runs.
  - A reference counts only by the prompt's agreement rule; those that do not are reported with
    their scatter and dropped. Fewer than five counted is **undecided**.

### Sequencing

1. **Now, while the rehearsals run:** read, copy the harness, write its diff, and plan Z0, the
   per-call records and the references. **Compute nothing** (no hashing of the store, no fingerprint,
   no unit run, no Z0): the rehearsals hold the six cores. End your turn reporting that you are
   ready, with the exact launch command you will use.
2. **When the orchestrator resumes you** (after R19, R20, the comparison and the readers are over,
   and the lock is released): check the store and 06c's checkpoint, run Z0, write
   `state/predictions.md`, then — **only if all three clauses of Z0 hold** — launch the census, make
   one check, and end your turn with the launch report. If Z0 fails, that is a stop (prompt §7):
   report it, and do not launch.
3. **Resumed with the census's final `status.json`:** run the references as above, one at a time,
   ending your turn at each launch.
4. **After the last reference:** score Z2–Z5 and the descriptive items, write your records, and
   report in full.

**Your final report** gives: the census run id, its `status.json`, its wall time and unit wall at
both rtols against 06c's (3.21 h and 5.77 h); Z0; Z2–Z5 each holds/fails/undecided with numbers;
the depth distribution and converged counts at both rtols against 06c's; over-budget units; the
references counted and dropped, with scatter; the store's and 06c's checkpoint's hashes before and
after; the run directories' sizes; your deviations; and anything in `issues.md`.
