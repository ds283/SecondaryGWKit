### B.6 Addendum: Part B's first stop, and the amendment it led to

*Written 2026-11-08, at the hold before the census.*

**Dispatch.** On the user's word (*"Please start Part B on the Mac Studio"*):
- §B.0's checks were repeated at 23:56 UTC on 2026-11-07, all read-only. The checkout was clean at
  `28ebaf07`, all 62 runs finished, the lock free with no Ray process, no Part B path, load about
  2.6 and 136 GiB free. The 160 input files matched their hashes.
- The laptop fetched and made `SecondaryGWKit-lpf07-91472db` at `91472dbe` and
  `SecondaryGWKit-lpf07-7460035` at `74600350`, detached, with no venv.
- It started the Mac Studio orchestrator session in the second. The marker was touched and 181
  files hashed before the first dispatch.

**The rehearsal agent, to stop 1 and past it:**
- **The rehearsals.** R27 (`levin-phase-floor-07-r27-parent-20261108T000634`) and R28
  (`-r28-guard-20261108T001916`) both give `1b332ffa…`.
- **The comparisons,** each checked by the Mac Studio orchestrator's own readers:
  - S0 holds: R27 equals R26 in all 143,831 `QuadSourceIntegral` fields and in every other value
    table, and the label differences are only the job name and the stamp.
  - Outside `QuadSourceIntegral`, R28 equals R27.
  - The 640 rows that located nothing are bit-identical.
  - 17 rows went from unconverged to converged, none the other way, and no leaf was guarded.
- **Stop 1:**
  - **The agent's question:** the store keeps only the counts, so how should it get each located
    point's kind and $z$, which §5 item 4, S1 and S2 need?
  - **The user:** *"I approve the replay. Also, add a reference to check 2 or 3 of the `q-r`
    calls, to settle whether R28 is honest. Hold Part B before the census."*
- **After stop 1:**
  - The replay reproduced all 470 calls that located a point, each bit for bit against R28.
  - S1 holds: 12 of 12 honest, the worst 0.218.
  - S2 holds: R28 at 0.020 and 0.263 of its error, R27 at $5.6\times10^5$ and $4.1\times10^4$.
  - The readers exit 0 and the lock was released.
- **The zero-rate case.** 26 `q-r` calls with $q = r$ located 4,097 points each. The three
  referenced show both stores honest (`[07-location-on-an-identically-zero-rate]`).
- **The census agent was not dispatched.**

**The user's rulings at the hold,** after the orchestrator explained why the C2 gate does not catch a
constant phase, what the integrand is there, the partly flat case and the resonance:
- *"Yes, rule A alone, please."*
- *"Open a separate issue for the `6pi` gate. We'll deal with this afterwards, as you say."*
- *"Record the resonance issues as belonging to the one-loop build and point at the planning
  document."*
- *"Please make the fix as an amendment of 07."*

The board's Decisions row is **07 Part B stop 1**. The amendment is prompt §10.

**Part A′ — the amendment's code, on the laptop.**
- **Dispatch.**
  - A fresh-context Opus subagent, in the main checkout at the amendment's commit.
  - Its scope is prompt §10, with §1, §3, §4 and §8 as they bear on it. It does not read
    `orchestrator/`.
  - Its gate is the prompt's (the stashes and `wip/` included), with `HEAD` at the amendment's
    commit and the seven roots' counts taken there (103, 781, 39, 156, 280, 127, 64 expected).
  - No planned stop. It stops at §8's conditions, and at §10's (a rung of §2 that moves; a test
    whose assertion moves).
- **The review, on the laptop, before landing:**
  1. **Scope.** The diff against the amendment's commit touches only:
     - `AdaptiveLevin/levin_quadrature.py`;
     - `AdaptiveLevin/tests/test_levin_stationary.py`;
     - `AdaptiveLevin/tests/test_levin_trust_yardstick.py`, the key set alone;
     - §2's new output directory under `docs/levin-phase-floor/07/calibration/out/`;
     - the log, the board and `docs/OPEN_ISSUES.md`.
  2. **The construction, by AST against `ad732887`.**
     - Only `_locate_stationary_points`, the guard's handling of a skipped phase, the code reporting
       the new key, and their docstrings change. Every other function is AST-equal.
     - Rule A compares against `SIX_PI` by name. Its estimate costs no evaluation beyond the scan.
  3. **§2's re-run.** Every call with no phase skipped is bit-identical to run 2. Re-run §2.2's
     124 units myself on the committed module, against run 2's outputs. The skipped calls are
     listed.
  4. **The tests.** Read test 13 (a)–(d) against §10. Re-apply breakages (x) and (xi) myself, each
     caught by its test. The seven roots rise only by the tests added.
  5. **The records.** The log's section is in README §5.1's form, with deviations numbered on from
     Part A's. The issue is moved to §4, and the index is corrected by counting.

  The code is pushed only on the user's word. A records commit then writes this note's §B.7, Part
  B's resumed dispatch, and its SHA is Part B's new dispatch SHA.

**Part B resumed, at the new dispatch SHA** (§B.7 gives the details):
- **The worktrees.** A third worktree, `SecondaryGWKit-lpf07-<new short SHA>`, made by the
  laptop after a fetch.
  - R27's worktree and R28's (`…-7460035`) are kept, with R28's store and records.
  - The Mac Studio orchestrator session goes on, given the new SHA by message. It reads this note
    and the prompt in the new worktree.
- **The rehearsal agent, resumed or fresh.**
  - **R29**, slug `r29-guard-rule-a`, at the new worktree, under the lock, with 06m's command
    line, giving `1b332ffa…`.
  - **R29 against R28:** they differ only in Levin calls whose phase Rule A skipped and which
    located a point at R28, and those calls equal R27's.
  - **R29 against R27:** as prompt §5 item 4.
  - **S1 and S2** re-scored on R29. S2's references from `p1-references` stand, being taken at
    R27's tree.
  - **The readers** on R29, then the lock released.
  - **Planned stop 1′:** before the census, to the user.
- **The census agent:** as §B.2, at the new worktree, with V3 as prompt §10 sets it. A phase
  Rule A skipped is set aside, its calls counted and listed.
- **Kept until the records commit:** R28's store, the `-7460035` worktree and the stop-1 records
  in `var/levin-phase-floor-07/`.

### B.7 Addendum: Part B's resumed dispatch

*Written 2026-11-08, at the amendment's landing. This addendum's commit is Part B's new dispatch
SHA.*

**The amendment landed.**
- Prompt §10 is `2ddd3306`. Its code is **`adffa21b`**, "Skip locating a phase whose variation is
  below the C2 gate". Both are pushed.
- `adffa21b` passed §B.6's five checks. The board's Decisions row is **07 amendment lands**.
  - **Scope:** only the allowed paths changed.
  - **By AST against `ad732887`:** only `_locate_stationary_points`, `_nodal_guard` (with the new
    `_changes_sign`), the threading of the skipped phases through `_adaptive_levin`,
    `_adaptive_levin_subregion` and `_impl`, and the result key `location_skipped_phases` change.
  - **§2.2:** 124 units re-run, bit-identical to run 2.
  - **Breakages (x) and (xi)** re-applied and caught.
  - **The seven roots:** 107, 781, 39, 156, 280, 127, 64, all OK.
- **The user:** *"Yes, please push both, then write and push §B.7, and send to the Mac Studio to be
  executed."*
- **§2 does not exercise Rule A.** No call of §2.2, §2.3 or §2.4 has a phase below $6\pi$, so all
  of §2 is bit-identical to run 2. Test 13 and R29 are where the rule is measured.
- **O9 is on `[07-c2-gate-threshold-was-never-measured]`:** a phase Rule A skips can still reach the
  Levin solve, because the gate's node-based estimate reads high. There the nodal guard covers it.

**The trees:**
- **The parent worktree** `SecondaryGWKit-lpf07-91472db` (`91472dbe`), unchanged. It runs V0's K0
  run.
- **The new code worktree** `SecondaryGWKit-lpf07-<this commit's short SHA>`, detached at this
  commit, with no venv.
  - Its code is `adffa21b`'s.
  - `git diff --stat 74600350 <this commit> -- . ':!docs' ':!prompts'` touches only
    `AdaptiveLevin/levin_quadrature.py`, `AdaptiveLevin/tests/test_levin_stationary.py`,
    `AdaptiveLevin/tests/test_levin_trust_yardstick.py` and `CLAUDE.md`.
  - It runs R29, the comparisons, S1 and S2's re-scoring, the readers, the census $g$ and V4's
    references.
  - The note and the prompt are read here.
- **The old code worktree** `SecondaryGWKit-lpf07-7460035` and R28's store are kept, read-only, as
  the record of the zero-rate case. No new run is made in that worktree.

**What the Mac Studio orchestrator does, in order.** It is the same session, told by message.
1. **Re-check before the lock.**
   - Every tree is clean, and each `HEAD` is as dispatched.
   - `runregistrykit list` shows nothing running or stale.
   - The lock is free, and `ray-processes` lists nothing; each is a command of its own.
   - Touch a new marker, and re-hash the 181 protected files against the first before-hashes.
2. **The rehearsal agent, resumed or fresh, on prompt §5 with §10 as written.**
   - **R29:**
     - slug `r29-guard-rule-a`, in the new worktree;
     - `--job-name levin-phase-floor-07 --campaign levin-phase-floor --prompt 07`;
     - 06m's command line;
     - a `--purpose` naming the host, the tree's SHA, the code commit `adffa21b` and the fingerprint
       `1b332ffa…`.
   - **The lock:** `acquire`, `set-pid` with R29's driver, held through the comparisons, the replay
     and the readers, then `release`. Each lock command is a shell command of its own.
   - **R29 against R28,** by the orchestrator's own reader, as well as the agent's:
     - outside `QuadSourceIntegral`, every value table identical;
     - inside it, a Levin record differs from R28's only if R28's count of located points was
       nonzero and R29's is 0. Every such record equals R27's in value, abserr, convergence and
       region count, bit for bit. The 26 `q-r` calls with $q = r$ are expected, and any other is
       listed;
     - every other Levin record bit-identical to R28's, its four counts included;
     - rows that differ hold such a record, and are tabulated.
   - **R29 against R27:** as prompt §5 item 4, with `tools.store_value_diff --json` and 05's
     `classify_non_qsi.py`, and the rows tabulated.
   - **The replay:** the calls of R29 that locate a point, by the harness the stop-1 replay used,
     run from the new worktree. Each is checked bit for bit against R29 before use. About 444 are
     expected: 470 less the 26.
   - **S1 and S2,** re-scored on R29. S2's references from `p1-references` stand, being taken at
     R27's tree.
   - **The readers** on R29: exit 0, and R29 byte-identical before and after. Then the lock is
     released, and `ray-processes` lists nothing.
   - **S0** stands: R27 is unchanged.
3. **Planned stop 1′, before the census, to the user, through the laptop.** Report:
   - R29's fingerprint;
   - R29 against R28 and against R27;
   - the replay, S1 and S2;
   - the readers and the lock;
   - R29's wall time and summed regions against R28's.
4. **The census agent, only on the user's word at stop 1′.** As §B.2, with the census tree $g$ the
   new worktree, V0's K0 run in the parent worktree, and V3 as prompt §10 sets it. Planned stops 2,
   3 and 4 as §B.2.
5. **§B.3's review** at the end, with the run checks taking in R29. R28 is checked unchanged, not
   re-reviewed.

**Unchanged:**
- git is read-only on the Mac Studio;
- `--cpus 6`, never `ray stop`, never another session's run;
- A3 v2, its scratch copy, 06m's census and R23–R26 are never written.
