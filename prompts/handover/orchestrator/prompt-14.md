# Orchestrator — prompt 14, B2a: give the realistic $T$ production's Liouville–Green construction

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code**, and the deliberate breakages are the agent's, not yours.

**The prompt:** [`prompts/handover/14-self-consistent-realistic-t.md`](../14-self-consistent-realistic-t.md)
**Board item:** B2a · **Closes, if its acceptance item 6 holds:**
`[13-the-tlg-term-depends-on-the-levin-subdivision]` · **Narrows:** `[00-tk-lg-truncation-floor]`
(`GkTk-remedial`), `[07-lg-derivative-truncation-at-handover]` (`source-remediation`) ·
**Model:** Opus, for both parts.

It runs as [`prompt-13.md`](prompt-13.md) ran B2: on the laptop, in a worktree of its own, with one
subagent continued across three stops. **§A**, Part A: the fixture, one commit; the agent stops and
you review. **§B**, Part B's launch: checks, `terms.md`, a registered run, one check, stop. You end
your turn. **§C**, Part B's records, after the run; you review and land.

## 0. The order, and why

- **The main checkout carries another session's work.** At writing, `levin-phase-floor` 07's
  changes to `AdaptiveLevin/levin_quadrature.py`, its test and `ComputeTargets/QuadSourceIntegral.py`
  were **staged** there, with untracked records. The worktree is cut from the committed dispatch
  SHA, so they are not in it. **Commit your own records by explicit path** (`git commit -- <paths>`),
  never with a bare `git commit` there, which would take the staged files with it.
- **The Levin driver has changed since B2** (`levin-phase-floor` 06b–06d). The prompt re-takes
  everything at the dispatch SHA and treats B2's numbers as context. If 07 has landed by dispatch,
  so much the better: record which driver ran.
- **The Mac Studio is not used.**

---

## A. Part A — the fixture

### A.0 Facts checked by the orchestrator

Pass them on.
- `docs/handover/b2/scripts/b2_cells.py` hashes `0a8ae0ebdf6ff072…`.
- B2's checkpoint is `var/runs/handover-13-residue-20261027T110832/checkpoint.jsonl` in the main
  checkout, read only.
- The line references hold at the planning commit:
  - `test_tk_source_functions.py`: `closed_form_residual` `:179`, `Fixture` `:233`,
    `exact_functions` `:434`, `LG_functions` `:483`, `test_omega_matches_phase_derivative`
    `:951`, `test_LG_truncation_error_of_the_exact_envelope` `:976`;
  - `test_tk_wkb_phase.py`: `build_and_store` `:243`, `_Shared.build_radiation` `:353`,
    `TestRadiationValue` `:489`;
  - `test_quadsource_integral.py`: `Case` `:301`, `production_fixture` `:318`,
    `exact_with_the_geometry_of` `:534`.
- **Why the route matters.** `build_and_store` drives production's `compute()` and `store()`
  offline, with `ray` mocked. It runs on `_Shared`'s radiation background, which is a real table
  model, not `Case`'s `FakeModel`. Whether `FakeModel` can drive it is for the agent to find out.
  The prompt allows two fallbacks, each with its own check.
- B2's smoke-test deviation (log 13, item 8) is why the prompt now forbids any ladder cell before
  `terms.md`. The user accepted it at B2, and does not want it repeated.

### A.1 Before you dispatch

1. Commit the prompt, this note and the board's B2a row, **by explicit path**. That commit is the
   dispatch SHA.
2. `git worktree add -b handover-14-b2a ../SecondaryGWKit-handover-14 <dispatch SHA>`; symlink
   `./venv`; `git status --short` empty there.
3. **Baselines:** the seven roots, serially, in the worktree, with `uptime`.
4. `touch` a marker in the scratchpad immediately before dispatch.

### A.2 Dispatch

One fresh-context subagent on **Opus**, given the prompt's path, the dispatch SHA, the worktree,
the baselines, and §0 and §A.0 of this note. It does not read `orchestrator/`. Tell it:
- **Work only in the worktree.** The main checkout is read only, for `runregistrykit list` and
  B2's checkpoint.
- **This turn is Part A only.** One commit on `handover-14-b2a`, staged by explicit path:
  test modules under `ComputeTargets/tests/`, `prompts/handover/logs/14-self-consistent-realistic-t.md`
  and `prompts/handover/IMPLEMENTATION_STATE.md`. Anything else is a stop. No push, merge or rebase.
- Stop and ask at any of prompt §5's conditions.

### A.3 The review of Part A — six checks, in the worktree

1. **Scope.** Only §A.2's paths; no production file in the diff.
2. **The route.** Read the construction. If it is production's producer, check that the captured
   values are `store()`'s. If it is a fallback, check that the equality check exists and passes.
   `exact_envelope_model()` is no longer what the realistic $T$ is read against.
3. **Test (a)'s number** is quoted, with its bound's provenance. Run it.
4. **The roots, re-run here:** six equal to the baselines; `ComputeTargets` up by exactly the tests
   added. The known flake re-run alone.
5. **Moved numbers** listed; no tolerance changed (read the diff for bounds and constants).
6. **The breakage record and the log.** Three breakages, each caught and reverted; deviations
   classified; the board row says "Part A landed" and moves no issue; `black --check` clean.

---

## B. Part B's launch

Continue the same agent: Part A passed; now prompt §2.1–§2.5. Remind it that **no ladder cell
runs before `state/terms.md` exists**. When it reports the launch, confirm from the registry
alone that the run `handover-14-residue-lg-*` is `running` with its pid alive and its heartbeat
fresh, and that `terms.md` is older than the manifest. **End your turn.** Tell the user the run
id and the estimate, and resume when they say so.

---

## C. Part B's records

When the user resumes you and the run is `done`, continue the agent for prompt §2.6 and §5's
records commit. Then review.

### C.1 The review — eight checks

1. **Scope:** `docs/handover/b2a/`, `docs/handover/SEAM-RESIDUE.md` (§8 added; §0–§7 unchanged,
   by `git diff`), the log, the three boards, `docs/OPEN_ISSUES.md`. No `.py` outside
   `docs/handover/b2a/scripts/`.
2. **Provenance:** the copied scripts' hashes equal the checkpoint's; every record carries Part A's
   SHA and an empty `git status --short`.
3. **`terms.md` first:** older than the manifest, and the log says what the scripts were tested on.
   No checkpoint record or state file shows a ladder cell before it.
4. **Completeness**, bit for bit, as saved.
5. **One cell re-run** at a cheap rung with the copied cell script: bit-identical.
6. **The arithmetic:** $D$, $A_i$, $B_i$ and both interactions recomputed from the checkpoint at
   `together` 980 and one top rung; compared with §8's tables. Every exception to $A_i = B_i$ is
   reported in §8.
7. **The issue movement matches the evidence.** `[13-…]` is closed only if acceptance item 6
   holds. The real-row question is settled by test (a)'s bound, as prompt §3 says.
8. **The records:** the log is complete; the narrowings are on their own boards, dated; the
   index count is right by counting. If the copied scripts are not `black`-clean, ask the user
   before listing `docs/handover/b2a/scripts/` in `CLAUDE.md`.

### C.2 Landing

As prompt 13's: rebase onto `handover-remedial` in the worktree, resolving a conflict on the
board or the index by reading both sides, and **re-counting the index after the rebase** (B2's
header count survived a clean rebase wrong). Re-run the roots if the rebase brought `.py` files
outside `docs/`. Then `git merge --ff-only` in the main checkout. **If the main checkout has staged
or modified files that a fast-forward would touch, stop and ask the user**: they are another
session's. Record the landed SHAs on the board, then remove the worktree and the branch. No push.
