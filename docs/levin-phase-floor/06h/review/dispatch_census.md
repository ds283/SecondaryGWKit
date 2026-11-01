## Your brief: the census agent — prompt 06h §5, the census (06e's census re-run under $h$; Gate H, H3, H4 and the descriptive items)

**Your scope is prompt 06h §5's census:**
- Gate H;
- the census run;
- H3 and H4;
- the descriptive items "What the declaration costs", "Convergence and depth", "Cost" and "The
  stationary-point rows".

Two parts of §5 are not yours:
- §5's **phase re-measurement** (H1 and H2) is the phase agent's;
- §4 (R21 and R22, under Ray and the lock) is the rehearsal agent's, and runs first.

**Where you write:**
- only `var/levin-phase-floor-06h/census/` (with `out/`, `scripts/` and `state/`) and your own run
  directories in `var/runs/`;
- you take no lock;
- no process of yours initialises Ray. Importing the `ray` package, as production modules do, is
  fine; a Ray session appearing is not.

**The harness** is 06e's census copy, from `docs/levin-phase-floor/06e/census/scripts/` in the code
worktree.
- Copy it into `var/levin-phase-floor-06h/census/scripts/`.
- Change only the tree and its paths (prompt §5).
- If the harness as copied cannot record something that Gate H, H3, H4 or a descriptive item needs,
  the change that records it is allowed. It is a numbered deviation, and it must not change any
  computed value.
- Save the diff against 06e's copy in `out/`. Each change is a numbered deviation.

**The trees:**
- Gate H's reproduction of 06e runs at the parent worktree (`2c325f5`);
- $h$ is the code worktree (`c1ec3b2`);
- 06e's census tree was `6d81fad`.

**The "before"** is 06e's census run `var/runs/levin-phase-floor-06e-census-20261030T081318/`,
read from its `checkpoint.jsonl` (`a0dc4d7f…`) and per-unit records.
- Read it only.
- Confirm its checkpoint's SHA-256 before and after.

**The store** is the scratch copy, opened read-only.
- Before the run, re-check it against its certification and fingerprint, with 06e's copied
  `fingerprint_copy.py` or its equivalent.
- Hash it (SHA-256) before and after, with no journal, `-wal` or `-shm` file beside it.

**Gate H runs before the launch,** both parts of prompt §5. Save its outputs in `out/`.
- **"The parent reproduces 06e."** Its git clause is against `6d81fad`. Take the list of modules
  the harness imports from a dry run at the parent. The reproduction of 06e's units `low-G+r-1` and
  `ctrl-low-1` at `1e-7`, and `0-1971` at `1e-8`, at the parent, is compared by bit pattern
  (`float.hex`).
- **"The new code is deterministic."** Two runs of one $h$ unit, compared by bit pattern.
- Choose the $h$ unit to be one with an analytic block and Levin calls whose groups supply
  `theta_abserr`. Say why it qualifies.

**The predictions** go in `state/predictions.md`, **after Gate H and before the launch**:
- prompt §5's H1–H4 text, unchanged (copied, not paraphrased);
- "the stationary-point rows" fixed as 06a's 720 (`spscan.py`), with how you identify them.

Its mtime is the review's evidence. Do not edit it after the launch; anything you add after the
launch goes in another file.

**What each record carries:**
- Every checkpoint record carries:
  - its script's SHA-256;
  - its tree's `HEAD` and `git status --short`;
  - each unit's evidence of where its modules were loaded from, so a reader can see that an $h$ unit
    imported from the code worktree and a parent unit from the parent worktree.
- Per call, record what H3, H4 and the descriptive items need, as 06e's census did:
  - the call's value, abserr, `converged`, regions and solves;
  - each returned leaf's depth and branch (Levin or Clenshaw–Curtis);
  - the declared phase term per leaf, summed per call by branch, as 06f's
    `declared_vs_request_06f.py` reads it;
  - the request.

  The orchestrator re-scores H3 and H4 from the two checkpoints with its own reader. State in your
  log where each field lives.

**The run:**
- **Workers:** at most six worker processes. A Ray session started by a census process, under
  `/tmp/ray-sgk` or anywhere, is a stop.
- **The budget** is 300 s per unit, enforced from the dispatcher, with 06a's over-budget pass, as
  06e ran it. Do not change the budget. Over-budget units are reported, and an over-budget unit
  whose Levin calls cannot be read counts against H4 as unresolved: say how you treat it.
- **Registration:** one registered, detached run with slug `census`
  (`--campaign levin-phase-floor --prompt 06h`) and job name `levin-phase-floor-06h`, launched as
  06e's was.
- **Every row at `1e-7` and at `1e-8`.** 06e's census was 18,560 units over 1 h 46 min of wall time,
  with unit wall 2.28 h at `1e-7` and 2.97 h at `1e-8`. A census much longer than 06e's is itself a
  finding.
- **Keep the checkpoint and the per-unit records (pickles) in full,** in your run directory and
  records. Do not prune them: **06g's census compares against them.** List them in
  `KEPT_ON_MAC_STUDIO.txt`.

**Scoring** (prompt §5), from the two checkpoints, each verdict holds, fails or undecided against its
rule as written, with its numbers:
- **H3:** every violating row, if any, with both totals and both `total_abserr` in `float.hex`.
- **H4:** "no Levin call that 06e reported converged is unconverged under $h$".
  - Say how calls are matched between the checkpoints: row, rtol and call index. Say what you do
    where the call lists differ in length or order.
  - List every call that lost convergence, and every call that gained it.
- **Descriptive:**
  - the declared term's sum over leaves per analytic-block call, under 06e and $h$, by branch and
    against the request;
  - converged calls and rows, `total_converged`, and the depth distribution, under 06e and $h$, at
    both rtols;
  - regions, solves and unit wall time against 06e;
  - how the 720 stationary-point rows moved.
  - If A3 v2 reaches the near-singular class (see the addendum and the code log), report it as such.

**Sequencing:**
1. **Now, while the rehearsals run:** read, copy the harness, write its diff, and plan Gate H and
   the per-call records. **Compute nothing:** no hashing of the store, no fingerprint, no unit run,
   no Gate H. The rehearsals hold the six cores. End your turn reporting that you are ready, with the
   exact launch command you will use.
2. **When the orchestrator resumes you,** after R21, R22, the comparison and the readers are over
   and the lock is released:
   1. check the store and 06e's checkpoint;
   2. run Gate H;
   3. write `state/predictions.md`;
   4. **only if both parts of Gate H hold,** launch the census, make one check, and end your turn
      with the launch report.

   If Gate H fails, that is a stop (prompt §7): report it, and do not launch.
3. **When you are resumed with the census's final `status.json`:**
   1. hash the store and 06e's checkpoint again;
   2. score H3, H4 and the descriptive items;
   3. write your records;
   4. report in full.

   The phase agent's stage follows yours, so do not leave a worker running.

**Your final report** gives:
- the census run id, its `status.json`, and its wall time and unit wall at both rtols against
  06e's;
- Gate H;
- H3 and H4, each holds, fails or undecided, with numbers;
- the declared term's sum against the request;
- the depth distribution and converged counts at both rtols against 06e's;
- over-budget units;
- the store's and 06e's checkpoint's hashes before and after;
- the run directory's size;
- your deviations;
- anything in `issues.md`.
