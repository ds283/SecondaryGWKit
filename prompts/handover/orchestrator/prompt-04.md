# Orchestrator — prompt 04, the $G_k$ phase decision test

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code.**

**The prompt:** [`prompts/handover/04-gk-phase-decision-test.md`](../04-gk-phase-decision-test.md)
**Board item:** A4 · **Closes:** nothing. It **narrows** two issues and decides whether A2's
numbers stand.

**Refreshed 2026-10-25** with the prompt, which had never run. What changed: the agent works in a
**worktree at `98e68d3`** on the laptop, not at `HEAD` (prompt header); it runs two reproduction
cells beside the six; every cell records `max_depth`, the region count per group and the
phase-limited count; the run is registered; and landing is a rebase, as B1's was. **The threshold
is unchanged.**

## 0. What makes this prompt unusual

It has exactly one failure mode that matters, and a green suite cannot detect it: **the agent
choosing the threshold for "material" after it has seen the numbers.** Every other check here is
routine. This one is not, and it is not a matter of the agent's good faith — by the time anyone
writes a verdict they have already seen the result.

The prompt therefore fixes the threshold itself and forbids the agent to change it. Your job is to
confirm the document carries it **verbatim** and that the verdict follows from it mechanically. If
the document argues its way around the threshold, or restates it in softer words, that is a stop
however reasonable the argument. The same goes for §2.1's new fields: they are **recorded beside**
the verdict and must not become a third condition.

The second thing to watch: **"no change" is the likely result and is a good one.** A2 already
measured the representation term as flat in $x$ rather than growing like $h^4x/384$, which is weak
evidence against contamination. An agent that returns "no material change" has done the job. Do not
treat a null result as a thin deliverable.

**It runs in parallel with two others** (the user, 2026-10-25): `handover` A3 and
`levin-phase-floor` 04, both on the Mac Studio. Neither touches the laptop's files or its registry,
and A4 touches nothing on the Mac Studio. The only contact is the board and `docs/OPEN_ISSUES.md`,
which A3's bookkeeping commit also edits. Whichever lands second resolves the conflict by reading
both sides.

---

## 1. Before you dispatch

1. **Confirm the branch and `HEAD`.** The campaign runs on **`handover-remedial`**. Record the SHA
   of the dispatch commit (this note, committed and pushed); the agent must be told it and told that
   `HEAD` is **not** its own.
2. **The worktree.** `git worktree add -b handover-04-a4 ../SecondaryGWKit-handover-04 98e68d3`;
   symlink `./venv` to the main checkout's (the common `info/exclude` already lists `venv`);
   `git -C ../SecondaryGWKit-handover-04 status --short` is empty.
3. **Take the baselines in the worktree** and record them: the seven roots, serially, at
   `98e68d3`, into the session scratchpad. This prompt adds no test, so they are a **ceiling and a
   floor**. (The 552 / 39 / 148 this note once carried were taken at `0d7c05c` and are stale.)
   **The known flake** (`test_tk_wkb_phase … test_wall_time_per_object`) is re-run alone and
   recorded; it is not a stop.
4. **Record A2's checkpoint identity before anything runs**, in the main checkout:
   ```bash
   wc -l < var/runs/realistic_large_x_cells.jsonl        # expect 60
   ./venv/bin/python -c "import json;print({json.loads(l)['script_sha256'][:12] for l in open('var/runs/realistic_large_x_cells.jsonl') if l.strip()})"
   shasum -a 256 var/runs/realistic_large_x_cells.jsonl
   ```
   Expect 60 cells and the single script hash `c1cd3598c23a`. **Record all three.** The agent must
   not modify this file; you check it afterwards. Confirm that the worktree's
   `docs/handover/realistic_large_x.py` hashes to `c1cd3598…` too.
5. **Take the comparison baseline yourself.** Recompute, from A2's checkpoint, the representation
   term for the four `together` cells at $x_{\rm resp} = 10^7$ and $10^8$ and the two `q-smooth`
   cells at $10^5$, together with `Levin_regions`, `integral_time`, `total` and the declared error
   for each. The representation term is $N_{\rm realistic} - N_{\rm exact}$ at fixed seam and $x$,
   computed through the block's common reference so it cancels. **Keep your numbers.** Checking the
   agent's "before" column against the agent's own arithmetic is not a check — this is the same
   reason prompt 02's control table was re-taken independently.
6. **The registry and the load.** `runregistrykit list` in the main checkout shows nothing
   `running` or stale. Record `uptime`. If anything heavy is running on the laptop, see §2's warning
   about timings.
7. **The marker:** `touch` a file in the scratchpad immediately before dispatch.

## 2. Dispatch

One fresh-context subagent, on **Opus**. Give it the prompt file and only that prompt, the dispatch
SHA, the worktree's path, the baselines, A2's checkpoint identity, and the files its "Read first"
list names.

Tell it plainly:
- **Work in the worktree.** Every command runs there with `PYTHONPATH=.`. In the main checkout it
  reads A2's checkpoint, by absolute path, and its run directory is created there by the registry;
  it writes nothing else there.
- **One commit, at the end, on `handover-04-a4`**, staged by explicit path:
  `docs/handover/gk_phase_decision_test.py`, its launcher if it is a separate file,
  `docs/handover/GK-PHASE-DECISION-TEST.md`, `prompts/handover/logs/04-gk-phase-decision-test.md`,
  `prompts/handover/IMPLEMENTATION_STATE.md` and `docs/OPEN_ISSUES.md`. **No push, no merge, no
  rebase:** landing is yours (§3a). Campaign README §3 already carries A4 and must not be touched.
- `black`; **suite counts unchanged, not risen**.
- **Launch the run registered and detached, verify once, and end its turn** rather than polling.
- §7's stop conditions mean *stop and ask*.

**Restate the threshold in the dispatch, verbatim**, so it is fixed by you as well as by the prompt:

> A change in the representation term is material if it exceeds the sum of the two cells' declared
> errors **and** changes the clamp-to-representation ratio by more than a factor of two. A cost
> change is material if `Levin_regions` moves by more than 2×.

**And restate the reproduction rule**, also fixed in advance: the cost verdict's "old" column for
`q-smooth` is the reproduction in this venv, with A2's record beside it; a reproduction `total`
outside its declared error of A2's is a stop.

**Warn it about machine load.** `Levin_regions` is load-independent and is the sound cost statistic;
`integral_time` is not, and a wall-clock comparison taken against a saturated machine is worthless.
If anything comparable is running on the laptop, the cost verdict rests on region counts, with
timings reported and explicitly marked as taken under load.

## 3. The review — eight checks, in the worktree

1. **The threshold is in the document verbatim**, before the tables, in the prompt's words. Not
   paraphrased, not softened, not supplemented with a third condition that lets a material result
   read as immaterial.
2. **The verdict follows from the threshold mechanically**, and is given **separately for the two
   issues**. They can come apart: the cost could collapse while the physics stands, or the reverse.
   A single merged verdict is a stop. §2.1's depth fields appear beside it and do not decide it.
3. **A2's checkpoint is intact.** 60 cells, one script hash `c1cd3598c23a`, the file's SHA-256 as you
   recorded it. If the agent edited `realistic_large_x.py`, every cell would have been discarded and
   silently recomputed under a different script — check the hash set has exactly one member.
4. **The "before" column matches the numbers you took in §1.5**, to the digits. A disagreement is a
   stop, not a rounding discussion. **The reproduction cells** are set against A2's record, and
   each `total` is inside its declared error of A2's.
5. **The run.** `runregistrykit list` shows it `done`, recorded at `98e68d3`. Every record in its
   checkpoint carries the script's SHA-256 (the committed script's) and the git SHA. Six, eight or
   ten cells, and if ten, the log says the `q-smooth` cost collapsed and why that licensed $10^6$.
6. **Suites read exactly the baselines you took in the worktree.** A rise means a test was added,
   which prompt §5 forbids.
7. **Scope.** `git show --stat handover-04-a4`: only §2's paths.
   **`docs/handover/realistic_large_x.py` and `REALISTIC-LARGE-X.md` are untouched** — this prompt
   does not correct A2, it decides whether A2 needs correcting.
   `ComputeTargets/tests/test_phase_groups.py` is untouched: fixing the false claim at `:301`
   belongs to B2. `find <main checkout>/var -newer <marker>` lists only the run directory.
8. **Bookkeeping.** Both `02-` issues **narrowed**, not closed — closing them belongs to whoever
   acts on the verdict. The A4 row and item on the board. The log records the worktree and the tree
   as `STRUCTURALLY REQUIRED` deviations. `black --check` clean. Commit message in `CLAUDE.md`'s
   form.

## 3a. Landing

In the worktree: `git rebase handover-remedial`. The board's A4 row and item will conflict with the
refresh's: take the agent's. Any other conflict on the board or the index is resolved by reading
both sides. The rebase brings no `.py` file outside `docs/` into question, because the commit adds
only `docs/` scripts; if it does, re-run the seven roots at the rebased tip. Then, in the main
checkout, `git merge --ff-only handover-04-a4`, and push. The landed SHA goes on the board in your
residue commit. Remove the worktree and the branch once landed.

After landing, `docs/handover/gk_phase_decision_test.py` is a record of its own SHA: at `HEAD` it
raises, through `realistic_large_x.py`. That is expected and is in its docstring.

## 4. What a good outcome looks like

- The threshold quoted before the numbers, and a verdict that reads as arithmetic against it.
- Two verdicts, one per issue, that may disagree with each other.
- If the result is null: A2's §4 and §7 stand as measured, both issues narrow to "the fixture and
  `test_phase_groups.py:301` are stale", and B2 inherits a documentation fix rather than a re-take.
  The depth fields then say which of the cost issue's other two mechanisms is still standing.
- If the cost collapses but the physics does not move: the `q-smooth` ceiling was an artefact, the
  instrument reaches further than A2 reported, and the representation term is still sound. That is
  the most interesting outcome and the one most likely to be under-reported — make sure the
  document says what it implies for A2 §8 item 5.

## 5. Stop and ask the user

Relay verbatim; do not adjudicate.

- **The representation term moves materially.** A2 §4 was taken on a contaminated fixture and
  re-taking it is the user's decision, not a follow-on the agent may start.
- **A reproduction cell's `total` is outside its declared error of A2's.**
- The `PrimitivePhase` cannot be built offline.
- The corrected fixture will not drive at $x = 1.6\times10^8$.
- The agent proposes editing `realistic_large_x.py`, `REALISTIC-LARGE-X.md`, `test_phase_groups.py`,
  `AdaptiveLevin/` or any production file.
- The agent proposes adjusting the threshold or the reproduction rule, for any reason.
- Any check in §3 fails. **Report it; do not repair it.**

## 6. After it lands

Report: the landed commit; the suite counts before and after; the threshold as the document states
it; the representation term and `Levin_regions` before and after for every cell, with your own §1.5
numbers beside the agent's; the reproduction against A2's record; the two verdicts; the depth
record's headline; A2's checkpoint cell count, script hash and SHA-256 before and after; and the
narrowing notes.

Then stop. What follows from the verdict — a re-take of A2, or a documentation fix folded into B2 —
is the user's call.
