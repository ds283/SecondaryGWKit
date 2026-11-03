## Your brief: the census agent — prompt 06g §5, the census (06h's census re-run under $g$; Gate V, V2, V5–V8 and the descriptive items)

**Your scope is prompt 06g §5's census:**
- Gate V;
- the census run;
- V2, V5, V6, V7 and V8;
- the descriptive items "The values", "Slow leaves", "The node sum", "The exit", "Cost" and "The
  stationary-point rows".

Two parts of §5 are not yours:
- §5's **references** (the captures, the Filon references, the comparison with 06i's 13, V3, V4
  and the descriptive item "The correction $-\sum E$") are the references agent's. It reads your
  checkpoint afterwards, so record per call what it needs to match a call: row, rtol, call index,
  value, abserr, $\sum E$ and whether the call has a Clenshaw–Curtis leaf;
- §4 (R23 and R24, under Ray and the lock) is the rehearsal agent's, and runs first.

**Where you write:**
- only `var/levin-phase-floor-06g/census/` (with `out/`, `scripts/` and `state/`) and your own run
  directories in `var/runs/`;
- you take no lock;
- no process of yours initialises Ray. Importing the `ray` package, as production modules do, is
  fine; a Ray session appearing is not.

**The harness** is 06h's census copy, from `docs/levin-phase-floor/06h/census/scripts/` in the code
worktree.
- Copy it into `var/levin-phase-floor-06g/census/scripts/`.
- Change the tree and its paths, and add what prompt §5 says it records:
  - per call, beside 06h's fields: $\sum E$ (`node_error_signed_sum`), $\sum|E_i|$
    (`node_error_abs_sum`), `node_limited`, and the round-off exit's firings and premise checks;
  - in its region records, per checked leaf: $r$, $\Delta_1$, $\phi_P$, trust and the returned
    value's kind (`used_interval`'s `convergence_ratio`, `delta1`, `phase_err`, `trusted`,
    `value_kind`, and what else the descriptive items need, such as `value_own`, `budget_err` and
    the node error $E$).
- **The premise wrapper** must pass `split_could_meet_budget=` through to
  `_roundoff_exit_premise_holds` (the code log's Observation 6). Check that the wrapped function's
  signature at $g$ is what your wrapper forwards, and that at the parent the wrapper still works.
- If the harness as copied cannot record something that Gate V, V2, V5–V8 or a descriptive item
  needs, the change that records it is allowed. It is a numbered deviation, and it must not change
  any computed value.
- Save the diff against 06h's copy in `out/`. Each change is a numbered deviation.

**The trees:**
- Gate V's reproduction of 06h runs at the parent worktree (`4f4f9ed`);
- $g$ is the code worktree (`1c5336e`);
- 06h's census tree was `c1ec3b2`.

**The "before"** is 06h's census run `var/runs/levin-phase-floor-06h-census-20261101T005037/`,
read from its `checkpoint.jsonl` (`3104c18e…`) and per-unit records.
- Read it only.
- Confirm its checkpoint's SHA-256 before and after.

**The store** is the scratch copy, opened read-only.
- Before the run, re-check it against its certification and fingerprint, with 06h's copied
  `fingerprint_copy.py` or its equivalent.
- Hash it (SHA-256) before and after, with no journal, `-wal` or `-shm` file beside it.

**Gate V runs before the launch,** all three parts of prompt §5. Save its outputs in `out/`.
- **"The parent reproduces 06h."** The reproduction of 06h's units `low-G+r-1` and `ctrl-low-1` at
  `1e-7`, and `0-1971` at `1e-8`, at the parent, compared with 06h's checkpoint by bit pattern
  (`float.hex`) on every recorded value: the row's `total`, `total_abserr`, the oracle, and every
  call's value, abserr, regions, depth and convergence. Take the list of modules the harness
  imports from a dry run at the parent.
- **"The integrand is 06h's."** `git diff --stat c1ec3b2 3ae6f3a -- . ':!docs' ':!prompts'`, and
  its file list against the rule.
- **"The new code is deterministic."** Two runs of one $g$ unit, compared by bit pattern. Choose
  the unit to be one with Clenshaw–Curtis leaves (so $\sum E$ is non-zero) and trusted leaves
  returning their quarters. Say why it qualifies.

**The predictions** go in `state/predictions.md`, **after Gate V and before the launch**:
- prompt §5's V2–V8 text, unchanged (copied, not paraphrased);
- "the stationary-point rows" fixed as 06a's 720 (`spscan.py`), with how you identify them;
- how calls are matched between the checkpoints (row, rtol and call index), and what you do where
  the call lists differ in length or order;
- how you will classify a lost call's cause for V8 — "(b) on a non-noise leaf", "`node_limited`",
  or "other" — from which recorded fields.

Its mtime is the review's evidence. Do not edit it after the launch; anything you add after the
launch goes in another file.

**What each record carries:**
- Every checkpoint record carries:
  - its script's SHA-256;
  - its tree's `HEAD` and `git status --short`;
  - each unit's evidence of where its modules were loaded from, so a reader can see that a $g$ unit
    imported from the code worktree and a parent unit from the parent worktree.
- Per call, what V2, V5–V8 and the descriptive items need, as 06h's census did, with the new fields
  above. State in your log where each field lives. The orchestrator re-scores V2 and V5–V8 from the
  two checkpoints with its own reader.

**The run:**
- **Workers:** at most six worker processes. A Ray session started by a census process, under
  `/tmp/ray-sgk` or anywhere, is a stop.
- **The budget** is 300 s per unit, enforced from the dispatcher, with 06a's over-budget pass, as
  06h ran it. Do not change the budget. Over-budget units are reported; an over-budget unit counts
  as false for V5 (prompt §5), and for V8 say how you treat one whose calls cannot be read.
- **A unit that ends in an error** (not over budget) is a stop (prompt §7).
- **Registration:** one registered, detached run with slug `census`
  (`--campaign levin-phase-floor --prompt 06g`) and job name `levin-phase-floor-06g`, launched as
  06h's was.
- **Every row at `1e-7` and at `1e-8`.** 06h's census was 18,560 units over 2 h 33 min of wall
  time, with unit wall 2.22 h at `1e-7` and 2.92 h at `1e-8`. On the laptop's draws 06g raised
  regions by about 20 %. A census much longer than 06h's is itself a finding, not an argument.
- **Keep the checkpoint and the per-unit records (pickles) in full,** in your run directory and
  records. Do not prune them: **07's census compares against them**, and the references agent reads
  them. List them in `KEPT_ON_MAC_STUDIO.txt`.

**Scoring** (prompt §5), from the two checkpoints, each verdict holds, fails or undecided against its
rule as written, with its numbers. **V8 first, and then stop to report (below).**
- **V8:** list every call outside the stationary-point rows that 06h reported converged and $g$
  does not, at each rtol, with its row, call index, block or group, both abserrs, the request,
  regions and depth, and its cause by your pre-fixed classification. Count the **rows** they fall in,
  by cause. Also list every call that gained convergence.
- **V2:** every violating row, if any, on `total` and on the oracle, with both values and both
  errors in `float.hex`; the largest ratio on each.
- **V5:** every call outside the stationary-point rows that reaches `depth_max` under $g$, and every
  over-budget unit, at each rtol.
- **V6:** `g-1e-8-0-33995` call 5: its ending (converged, `node_limited` or neither), depth,
  regions, $\sum E$ and $\sum|E|$, under 06h and $g$.
- **V7:** `g-1e-8-3-1072` call 13 (06h's H4 call): converged or not, abserr, request, regions,
  depth, and the exit's firings and premise checks, under 06h and $g$.
- **Descriptive:**
  - the values: per call, $|v_g - v_{06h}|$ against ${\rm abserr}_g$, split by cause (trusted leaves
    returning $Q$, $\sum E$ in the error, a different partition);
  - slow leaves: the counts of the code log's test 3 classes over the census ($r < 1/4$; $1/4 \le r
    < 1$ at or below the floor; $1/4 \le r < 1$ above it; $r \ge 1$), and the regions and solves
    they add;
  - the node sum: per call, $\sum|E|/|\sum E|$ where $\sum|E|$ is at least 1 % of the call's error;
  - the exit: firings and premise checks under 06h and $g$; calls whose exit changed;
  - cost: regions, solves and unit wall time against 06h; the depth distribution and converged
    counts at both rtols;
  - how the 720 stationary-point rows moved (07's, not scored);
  - if A3 v2 reaches the near-singular class (see the addendum and 06h's census
    `nearsingular06h.py`), report it as such.

**Sequencing:**
1. **Now, while the rehearsals run:** read, copy the harness, make and diff your changes, and plan
   Gate V and the per-call records. **Compute nothing:** no hashing of the store, no fingerprint, no
   unit run, no Gate V. The rehearsals hold the six cores. End your turn reporting that you are
   ready, with the exact launch command you will use.
2. **When the orchestrator resumes you,** after R23, R24, the comparison and the readers are over
   and the lock is released:
   1. check the store and 06h's checkpoint;
   2. run Gate V;
   3. write `state/predictions.md`;
   4. **only if all three parts of Gate V hold,** launch the census, make one check, and end your
      turn with the launch report.

   If Gate V fails, that is a stop (prompt §7): report it, and do not launch.
3. **When you are resumed with the census's final `status.json`:**
   1. hash the store and 06h's checkpoint again;
   2. score V8 only, write it to `out/v8.md` with its table, and **end your turn** reporting V8's
      lost calls and rows by cause. The orchestrator decides whether the note's stop on V8 applies
      before the rest is scored.
4. **When resumed again:** score V2, V5, V6, V7 and the descriptive items; write your records;
   report in full. The references agent's stage follows yours, so do not leave a worker running.

**Your final report** gives:
- the census run id, its `status.json`, and its wall time and unit wall at both rtols against
  06h's;
- Gate V, all three parts;
- V2 and V5–V8, each holds, fails or undecided, with numbers;
- the depth distribution and converged counts at both rtols against 06h's;
- over-budget units;
- the store's and 06h's checkpoint's hashes before and after;
- the run directory's size;
- where in your checkpoint the references agent finds each call's value, abserr, $\sum E$ and
  Clenshaw–Curtis flag;
- your deviations;
- anything in `issues.md`.
