# Orchestrator — prompt 04, the $G_k$ phase decision test

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code.**

**The prompt:** [`prompts/handover/04-gk-phase-decision-test.md`](../04-gk-phase-decision-test.md)
**Board item:** A4 · **Closes:** nothing. It **narrows** two issues and decides whether A2's
numbers stand.

## 0. What makes this prompt unusual

It has exactly one failure mode that matters, and a green suite cannot detect it: **the agent
choosing the threshold for "material" after it has seen the numbers.** Every other check here is
routine. This one is not, and it is not a matter of the agent's good faith — by the time anyone
writes a verdict they have already seen the result.

The prompt therefore fixes the threshold itself and forbids the agent to change it. Your job is to
confirm the document carries it **verbatim** and that the verdict follows from it mechanically. If
the document argues its way around the threshold, or restates it in softer words, that is a stop
however reasonable the argument.

The second thing to watch: **"no change" is the likely result and is a good one.** A2 already
measured the representation term as flat in $x$ rather than growing like $h^4x/384$, which is weak
evidence against contamination. An agent that returns "no material change" has done the job. Do not
treat a null result as a thin deliverable.

---

## 1. Before you dispatch

1. **Confirm the branch and `HEAD`.** The campaign runs on **`handover-remedial`**. Record the SHA;
   the agent must be told it and told that `HEAD` is **not** its own. Prompts 01 and 02 landed at
   `c414451` and `0d7c05c`.
2. **Take the baselines** and record them. At `0d7c05c` they were `ComputeTargets` **552**,
   `CosmologyModels` **39**, `LiouvilleGreen` **148** (`skipped=1`), all OK. This prompt adds no
   test, so these are a **ceiling and a floor**.
3. **Record A2's checkpoint identity before anything runs**:
   ```bash
   wc -l < var/runs/realistic_large_x_cells.jsonl        # expect 60
   ./venv/bin/python -c "import json;print({json.loads(l)['script_sha256'][:12] for l in open('var/runs/realistic_large_x_cells.jsonl') if l.strip()})"
   ```
   Expect 60 cells and the single script hash `c1cd3598c23a`. **Record both.** The agent must not
   modify this file; you check it afterwards.
4. **Take the comparison baseline yourself.** Recompute, from A2's checkpoint, the representation
   term for the four `together` cells at $x_{\rm resp} = 10^7$ and $10^8$ and the two `q-smooth`
   cells at $10^5$, together with `Levin_regions` and `integral_time` for each. The representation
   term is $N_{\rm realistic} - N_{\rm exact}$ at fixed seam and $x$, computed through the block's
   common reference so it cancels. **Keep your numbers.** Checking the agent's "before" column
   against the agent's own arithmetic is not a check — this is the same reason prompt 02's control
   table was re-taken independently.
5. **Check whether a datastore pilot or other heavy job is running.** `var/runs/*/run.pid`, and
   `ps`. If one is, see §2's warning about timings.
6. `git status` clean.

## 2. Dispatch

One fresh-context subagent. Give it the prompt file and only that prompt, the SHA, the baselines,
A2's checkpoint identity, and the files its "Read first" list names.

Tell it plainly: **one commit**; log at `logs/04-gk-phase-decision-test.md`; board, campaign README
§3's A4 row and `docs/OPEN_ISSUES.md` in the same commit; `black`; **suite counts unchanged, not
risen**; launch anything long **detached and end its turn** rather than polling; and that §7's stop
conditions mean *stop and ask*.

**Restate the threshold in the dispatch, verbatim**, so it is fixed by you as well as by the prompt:

> A change in the representation term is material if it exceeds the sum of the two cells' declared
> errors **and** changes the clamp-to-representation ratio by more than a factor of two. A cost
> change is material if `Levin_regions` moves by more than 2×.

**Warn it about machine load if a pilot is running.** `Levin_regions` is load-independent and is
the sound cost statistic; `integral_time` is not, and a wall-clock comparison taken against a
saturated machine is worthless. If the A3 datastore pilot or anything comparable is running, the
cost verdict must rest on region counts, with timings reported and explicitly marked as taken under
load.

## 3. The review — seven checks

1. **The threshold is in the document verbatim**, before the tables, in the prompt's words. Not
   paraphrased, not softened, not supplemented with a third condition that lets a material result
   read as immaterial.
2. **The verdict follows from the threshold mechanically**, and is given **separately for the two
   issues**. They can come apart: the cost could collapse while the physics stands, or the reverse.
   A single merged verdict is a stop.
3. **A2's checkpoint is intact.** 60 cells, one script hash `c1cd3598c23a`, file unchanged. If the
   agent edited `realistic_large_x.py`, every cell would have been discarded and silently
   recomputed under a different script — check the hash set has exactly one member.
4. **The "before" column matches the numbers you took in §1.4**, to the digits. A disagreement is a
   stop, not a rounding discussion.
5. **Suites read exactly 552 / 39 / 148 (skipped=1).** A rise means a test was added, which prompt
   §5 forbids.
6. **Scope.** `git diff --name-only HEAD~1 HEAD`. Permitted: `docs/handover/gk_phase_decision_test.py`,
   `docs/handover/GK-PHASE-DECISION-TEST.md`, `prompts/handover/IMPLEMENTATION_STATE.md`,
   `prompts/handover/README.md` (the A4 row), `prompts/handover/logs/04-*.md`,
   `docs/OPEN_ISSUES.md`. **`docs/handover/realistic_large_x.py` and `REALISTIC-LARGE-X.md` must be
   untouched** — this prompt does not correct A2, it decides whether A2 needs correcting.
   `ComputeTargets/tests/test_phase_groups.py` must be untouched: fixing the false claim at `:299`
   belongs to B2.
7. **Bookkeeping.** Both `02-` issues **narrowed**, not closed — closing them belongs to whoever
   acts on the verdict. A4 rows on the board and in campaign README §3. `black --check` clean.
   Commit message in `CLAUDE.md`'s form.

## 4. What a good outcome looks like

- The threshold quoted before the numbers, and a verdict that reads as arithmetic against it.
- Two verdicts, one per issue, that may disagree with each other.
- If the result is null: A2's §4 and §7 stand as measured, both issues narrow to "the fixture and
  `test_phase_groups.py:299` are stale", and B2 inherits a documentation fix rather than a re-take.
- If the cost collapses but the physics does not move: the `q-smooth` ceiling was an artefact, the
  instrument reaches further than A2 reported, and the representation term is still sound. That is
  the most interesting outcome and the one most likely to be under-reported — make sure the
  document says what it implies for A2 §8 item 5.

## 5. Stop and ask the user

Relay verbatim; do not adjudicate.

- **The representation term moves materially.** A2 §4 was taken on a contaminated fixture and
  re-taking it is the user's decision, not a follow-on the agent may start.
- The `PrimitivePhase` cannot be built offline.
- The corrected fixture will not drive at $x = 1.6\times10^8$.
- The agent proposes editing `realistic_large_x.py`, `REALISTIC-LARGE-X.md`, `test_phase_groups.py`
  or any production file.
- The agent proposes adjusting the threshold, for any reason.
- Any check in §3 fails. **Report it; do not repair it.**

## 6. After it lands

Report: the commit; the suite counts before and after; the threshold as the document states it; the
representation term and `Levin_regions` before and after for every cell, with your own §1.4 numbers
beside the agent's; the two verdicts; A2's checkpoint cell count and script hash before and after;
and the narrowing notes.

Then stop. What follows from the verdict — a re-take of A2, or a documentation fix folded into B2 —
is the user's call.
