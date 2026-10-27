# Orchestrator — prompt 13, B2: score the residue the gap was masking

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code**, and the deliberate breakages are the agent's, not yours.

**The prompt:** [`prompts/handover/13-score-the-unmasked-residue.md`](../13-score-the-unmasked-residue.md)
**Board item:** B2 · **Closes:** `[02-realistic-fixture-Gk-phase-is-the-superseded-construction]`,
`[04-table-cs-tau-fixture-does-not-cover-z-init-when-the-first-sample-is-dropped]`,
`[12-realistic-large-x-reads-the-retired-clamp-records]` · **Narrows:**
`[04-production-phase-gk-tau-table-has-a-zero-width-bottom-interval]`; on their own boards
`[05-…]`, `[06-…]`, `[07-…]` (`source-remediation`) and `[00-tk-lg-truncation-floor]`
(`GkTk-remedial`) · **Model:** Opus, for both parts.

The whole prompt runs **on the laptop**, orchestrated from a laptop session, with one subagent
continued across three stops:
- **§A, Part A (prompt §1):** the fixtures. One commit; the agent stops; you review.
- **§B, Part B's launch (prompt §2):** the pre-launch checks and the registered run. The agent
  launches, checks once, and stops. You end your turn.
- **§C, Part B's records (prompt §3, §7):** when the run has ended. The document and the records
  commit; you review and land.

## 0. The order, and why

- **The Mac Studio is not used.** `levin-phase-floor` 06a holds it for a few hours from
  2026-10-27; B2 needs neither its store nor its Ray. Nothing here waits for it.
- **06a's records will land on `handover-remedial` from the laptop** while B2 runs. They touch
  `prompts/levin-phase-floor/`, `docs/levin-phase-floor/`, perhaps this board's §3 and
  `docs/OPEN_ISSUES.md`. So **B2 works in a worktree of its own** and lands by rebase; a conflict
  on the board or the index is resolved by reading both sides.
- **The worktree also isolates the run.** Each cell is a child process that imports from the
  worktree's tree when it starts. If `levin-phase-floor` 07, which changes `AdaptiveLevin/`, were
  committed in the main checkout mid-run, a run there would mix two drivers. In the worktree it
  cannot.
- **The laptop is loaded.** At writing, the 1-minute load was 21 on 10 cores (other projects'
  Ray, U2). The prompt caps the run at two cells at once and makes times indicative. The roots'
  wall times will be long; the counts are the signal.

---

## A. Part A — the fixtures

### A.0 Facts checked by the orchestrator at the dispatch SHA

Pass them on.
- **The registry** shows nothing `running` or stale on this host. 06a is registered on the Mac
  Studio and does not appear here.
- **`realistic_large_x.py`** hashes `c1cd3598c23a52051a0eb2695bf67977265383b5fa02977fb61d8251c2cd98a3`.
- **The line references** hold: `FakeQuadSource` `test_quadsource_integral.py:146`,
  `OffsetBesselPhaseGk` `:217`, `Case` `:340`, `production_seam_inputs` `:934`;
  `BesselPhaseGk` `test_phase_groups.py:300`; `_TableCsTauFixture` `test_g_phase_reanchor.py:448`,
  `_ProductionPhaseGk` `:664`; `_table_fixture_functions` `test_t_phase_reanchor.py:288`.
- **The clamp grep** outside `venv/`, `docs/` and `prompts/` finds the four stale comments the
  prompt names (`primitive_phase.py:113`, `:175`; `test_primitive_phase.py:416`; `main.py:1890`,
  found by its text) and the three `assertNotIn` checks in `test_quadsource_integral.py`
  (`:985`, `:987`, `:1146`), which stay.
- **At `HEAD` the fixtures already carry `theta_residual`** (`levin-phase-floor` 06):
  `_ProductionPhaseGk` forms it in double from the closed form, and `Fixture`'s samples carry
  `closed_form_residual`. So production's $G$ and $T$ can be built offline; the prompt's stop on
  "a fixture cannot be made production's" is not expected to fire.
- **The stationary-point expectation.** At $b=0$, $c_s$ is constant, $G$'s Bessel order is $1/2$
  (its phase exactly linear in $\eta$) and each $T$'s phase derivative approaches its large-$x$
  value from below. A group's $\theta'$ then has no zero in a Liouville–Green sub-interval unless
  $(q+r)c_s = k$ or $|q-r|c_s = k$ to within that approach. No A2 shape comes close:
  $(q+r)c_s/k$ is 1.15, 12.7 and 0.64 and $|q-r|c_s/k$ is 0.105, 1.155 and 0.63 for `together`,
  `T-first` and `q-smooth`. The nearest is `T-first`'s $G\pm q\mp r$, where $\theta'$ stays near
  $-0.16\,k$ at $T_r$'s hand-over and approaches $-0.155\,k$ deep inside.
  This is the orchestrator's reading; the agent measures it (prompt §2.5 item 6).

### A.1 Before you dispatch

1. Commit the prompt, this note and the board's B2 row; that commit is the **dispatch SHA**.
2. `git worktree add -b handover-13-b2 ../SecondaryGWKit-handover-13 <dispatch SHA>`; symlink
   `./venv` to the main checkout's (the common `info/exclude` lists `venv`);
   `git -C ../SecondaryGWKit-handover-13 status --short` is empty.
3. **Baselines:** the seven roots, serially, in the worktree, into the session scratchpad, with
   `uptime` beside them.
4. **The marker:** `touch` a file in the scratchpad immediately before dispatch.

### A.2 Dispatch

One fresh-context subagent on **Opus**. Give it the prompt's path, the dispatch SHA, the
worktree's path, the baselines, and §0 and §A.0 of this note. It does not read `orchestrator/`.
Tell it plainly:
- **Work only in the worktree.** Every command runs there with `PYTHONPATH=.`. The main checkout
  is read only for `runregistrykit list` and the run directory; any other write there is a stop.
- **This turn is Part A only** (prompt §1). One commit on `handover-13-b2`, staged by explicit
  path: test modules under `ComputeTargets/tests/`, the four comment edits
  (`ComputeTargets/primitive_phase.py`, `ComputeTargets/tests/test_primitive_phase.py`,
  `main.py`), `prompts/handover/logs/13-score-the-unmasked-residue.md` and
  `prompts/handover/IMPLEMENTATION_STATE.md`. A path outside that list is a stop. **No push, no
  merge, no rebase.** Then stop and report.
- **Stop and ask** at any of prompt §6's conditions, and relay nothing it was not asked.

### A.3 The review of Part A — seven checks, in the worktree

1. **Scope.** `git show --stat handover-13-b2`: only §A.2's paths. In the three production files
   the diff is comments only (`git diff <dispatch SHA> handover-13-b2 -- main.py
   ComputeTargets/primitive_phase.py`, read). `docs/handover/realistic_large_x.py` still hashes
   `c1cd3598…`.
2. **$G$ is production's.** `Case(exact=False)`'s $G$ phase is built by `_build_phase`; read the
   construction, then run test (a). The zero-width fix is one of the two the prompt allows, and
   says which.
3. **$T$ and the seam are production's.** No keyword to `TkSourceFunctions`; a table `cs_tau`
   covering $z_{\rm init}$ off a node; `drop_first_WKB_sample=True` in `Case`, and `Fixture`'s
   default unchanged (`git diff` on `test_tk_source_functions.py` is empty or leaves it).
4. **The roots, re-run here.** Six equal to the baselines; `ComputeTargets` up by exactly the test
   methods added, counted from the diff. All OK; the known flake re-run alone.
5. **Moved numbers.** Every realistic-flavour test whose number moved is listed with old and new;
   none had its tolerance changed (`git diff` read for `assertLess`/`assertAlmostEqual` bounds and
   tolerance constants).
6. **The breakage record.** Each of the three breakages names its failing test and was reverted.
   If a record is thin, re-apply it by hand in the worktree, run the module, and revert.
7. **The records.** The log's Part A half classifies every deviation and lists the records at
   their SHA (§1.5). The board's B2 row says "Part A landed" and moves no issue. `black --check`
   clean on every touched `.py` file. The commit message follows `CLAUDE.md`.

A failed check is a stop: report it to the user, as README rule 6 says.

---

## B. Part B's launch

Continue the **same agent** (SendMessage), telling it: Part A's review passed; now prompt §2 —
the script under the worktree's `var/handover-13/scripts/`, §2.5's checks saved under
`var/handover-13/state/`, `state/terms.md` last, then the registered launch, **one** check that it
started, and stop. If the agent is no longer reachable, dispatch a fresh one with the prompt, Part
A's commit and its log, and the same instructions.

When it reports the launch:
- confirm from the registry alone (`runregistrykit list`): the run `handover-13-residue-*` is
  `running`, its pid answers `kill -0` and its heartbeat is fresh;
- confirm `state/terms.md`'s mtime is older than the run's `manifest.json`;
- **end your turn.** Tell the user the run's id and the agent's estimate of its length, and that
  you resume when they say so. Do not poll.

---

## C. Part B's records, after the run

Resume when the user says so. `runregistrykit list` shows the run `done` (or not: a `failed` run
or one still `running` is reported to the user, not repaired). Continue the agent for prompt §3
and §7: the document, the copy of `var/handover-13/` back to `docs/handover/b2/` without the
checkpoint, and the records commit on `handover-13-b2`. Then review.

### C.1 The review of Part B — nine checks

1. **Scope.** The records commit touches only `docs/handover/b2/`, `docs/handover/SEAM-RESIDUE.md`,
   the log, the three boards (`handover`, `source-remediation`, `GkTk-remedial`),
   `docs/OPEN_ISSUES.md`, and `CLAUDE.md`'s formatting paragraph if the copied scripts are not
   `black`-clean. No `.py` file outside `docs/handover/b2/scripts/`.
2. **The scripts are what ran.** `sha256sum` of each copied script equals the hash on the run's
   checkpoint records; every record carries the Part A commit as `HEAD` and an empty
   `git status --short`.
3. **The terms were fixed first.** `state/terms.md`'s mtime precedes the manifest; its mapping is
   the one the cells record as delegated.
4. **Completeness, read.** The saved check shows E′ + all five ≡ R and R − all five ≡ E′ in
   `float.hex` of `total` and `total_abserr`. Disjointness is shown by listing.
5. **The controls.** E against A2's Table 2, the geometry control, and R against A4's arm P, as
   saved and as the document quotes them. The stationary-point scan reports no sign change.
6. **One cell, re-run.** Pick one realistic hybrid at a cheap rung and re-run it in the worktree
   with the copied cell script: `total` and `total_abserr` bit-identical to its checkpoint record.
7. **The arithmetic.** From the checkpoint, recompute $D$, the five $A_i$ and $B_i$ and both
   interactions at two rungs, `together` at 980 and one top rung, and compare with the document's
   tables. Check that every term carries its error, and that a rung reported additive meets prompt
   §2.3's rule.
8. **The document.** §0–§7 present; §7 names at least the prompt's list; earlier records are
   labelled with their trees and kept out of the measurement's columns.
9. **The records.** The log is complete and classifies every deviation. The three closed issues
   are in the board's §4 and gone from the index; the narrowings are on their own boards, dated;
   `docs/OPEN_ISSUES.md`'s count and date are right **by counting**. The commit message follows
   `CLAUDE.md`.

### C.2 Landing

In the worktree: `git rebase handover-remedial`, resolving a conflict on a board or the index by
reading both sides. If the rebase brought any `.py` file outside `docs/`, re-run the seven roots
there. Then, in the main checkout, `git merge --ff-only handover-13-b2`. Record the landed SHAs on
the board in a residue commit if the rebase changed them. Remove the worktree and the branch once
landed. Push only if the user asks.
