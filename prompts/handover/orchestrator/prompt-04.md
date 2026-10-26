# Orchestrator — prompt 04 (A4), does A2's Levin cost growth survive production's phases?

Read [`README.md`](README.md) in this directory first, and [`prompt-11.md`](prompt-11.md), whose
arrangement this note reuses. **You do not write code.**

**The prompt:** [`prompts/handover/04-gk-phase-decision-test.md`](../04-gk-phase-decision-test.md),
rewritten 2026-10-26 on the user's decision. **Board item:** A4 · **Closes:** nothing by itself.
Its result goes to the user, who decides whether
`[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]` closes · **Model:** Opus, as the prompt
recommends. You, the orchestrator, review on whatever model you were started with.

*(This note replaces the one written for the 2026-10-25 refresh, which dispatched A4 on the laptop
in a worktree at `98e68d3` and never ran. That version is in the git history, at `f1a3888`.)*

## 0. What makes this dispatch what it is

**You run on the Mac Studio, as 11's orchestrator did.** You are a Claude Code session over the
saved `MacStudio` SSH connection, in `/Users/ds283/Documents/Code/SecondaryGWKit`. You dispatch the
executing agent **in that checkout**, wait on its run, review there, and **commit nothing**. The
records commit is made on the laptop (§4).

**The run is long and Ray-free.** It has 60 cells, up to 4 at once, each with a 3-hour budget.
Expect a few hours. Most of it is arm S and `q-smooth` at $10^6$, which A2 estimated at hours per
cell. It takes no lock, so another Ray-free prompt, or a build that holds the lock, may share the
host. Neither may move the checkout while A4 runs.

**Facts checked by the laptop orchestrator on 2026-10-26.** Pass them on.
- **The Mac Studio**, read-only at 14:25 UTC:
  - `handover-remedial` at `149c393` before the fast-forward, clean, one worktree;
  - the registry holds 19 runs, all finished, and the lock is free;
  - there is no `var/handover-04/`;
  - `docs/handover/realistic_large_x.py` hashes `c1cd3598c23a…`, A2's;
  - **A2's checkpoint is not on this host:** `var/runs/realistic_large_x_cells.jsonl` exists only
    on the laptop. A2's curves are context only (prompt §1.1), and they are taken from
    `REALISTIC-LARGE-X.md` Table 8 and the board's entry;
  - 117 GiB free; the load is about 5.7–6.6 on 10 cores before A4.
- **On the laptop, at `abca7e9`, by a scratch script that built objects and ran one cell:**
  - `_ProductionPhaseGk(shape.k, w, case.z_resp, case.model, grid)` **builds** on the realistic
    `Case`'s own $G$ grid for `together` at 980 (168 nodes, down to $z_{\rm response}$). `.phase`
    is a `PrimitivePhase` with `exact_remainder` true, and `.old_phase` the same with it false.
    So the prompt §2.1 fallback, building without the `phase_spline` its `super().__init__` makes,
    was not needed there. It may be at other rungs;
  - **`partition["handover"]`** holds `Tq` and `Tr`, each with `z_init`, `z_switch` and
    `handover_continuity`. The sub-interval records hold `method`, `regime`, `z_max` and `z_min`,
    and no clamp field;
  - one cell as the realistic `Case` builds it at `HEAD`, which is arm S's $G$ with the fixture's
    old $T$, took **20.5 s** for `together` at 980 (A2: 18.5 s).
- **Shell.** Every command that takes arguments from a variable runs through `bash`, not zsh.

## 1. Before you dispatch

The laptop has already:
1. fetched and fast-forwarded the Mac Studio's checkout to the **dispatch SHA**, the commit that
   adds this note (`git merge --ff-only`);
2. started you.

Then you:
1. **The Gate**, re-read by you:
   - `git rev-parse HEAD` is the dispatch SHA, and `git status --short` is empty;
   - `runregistrykit list` shows nothing `running` or stale;
   - there is no `var/handover-04/`;
   - `realistic_large_x.py` hashes `c1cd3598…`.

   Anything else is a stop. Do not fetch.
2. **The marker:** `touch` a file in your session scratchpad immediately before dispatch.
3. **No suite baselines.** The prompt changes no tracked file.

## 2. Dispatch

One fresh-context subagent, on **Opus**, in the Mac Studio's checkout. Give it the prompt, the
dispatch SHA and §0 of this note. It does not read `orchestrator/`. Tell it plainly:
- **`HEAD` is not its own.**
- **Commit nothing, and run no git command that writes**, `fetch` included.
- **Write only under `var/handover-04/`**, its own run directory and launch log. A write anywhere
  else is a stop. That includes any tracked file, the store and other prompts' `var/` directories.
- **Save every command and its output.** The review reads the outputs, not the log.
- **The order is fixed by prompt §3:**
  - the construction check, the wrapper's transparency and determinism, saved;
  - then `state/readings.md`, prompt §4 copied unchanged;
  - then the launch.
- **Launch, one check, end the turn.** It reports the run id, the run directory, the detached
  Python pid, the `sh` pid and the first checkpoint line. You wait on the pid once, in the
  background, and resume the agent with the final `status.json`.
- **No reading is chosen after the numbers are seen.** Prompt §4 says what each reading means,
  and that "neither" is an answer. If the numbers support neither, the agent says so and gives
  them.
- **Every deviation is numbered and classified** before it reports, failed attempts included.
- **Stop and ask** at any of prompt §8's conditions.

## 3. The review — on the Mac Studio, after the agent reports

1. **Scope.**
   - `git status --short` is empty, and `HEAD` is the dispatch SHA.
   - `find var -newer <marker>` lists only `var/handover-04/`, the run's directory and its launch
     log, and any other session's own directory.
   - `realistic_large_x.py` still hashes `c1cd3598…`.
2. **The run.**
   - `runregistrykit list` shows it `done`.
   - Its manifest names `handover`, `04` and `cost-growth`, with `git_dirty: false` and `git_head`
     the dispatch SHA.
   - There are 60 cell records, each `ok` or `not reached`, and every `not reached` cell is over
     its 3-hour budget.
3. **The arms, read** against prompt §2.1, in the script:
   - **S** is `Case`'s own `OffsetBesselPhaseGk`, unchanged;
   - **P₀ and P** differ from S in `.phase` alone, and from each other only in `exact_remainder`;
   - **$T$ in S, P₀ and P** is a `_TableCsTauFixture` with `drop_first_WKB_sample=True`, built
     with no keyword;
   - **E** is `ExactGk` and `ExactTkFunctions`;
   - the tolerances are `REF_ATOL` and `REF_RTOL`, and the clamp reads are replaced by
     `partition["handover"]`.

   Then **independently, by a scratch reader that builds the objects and runs no integral,** for
   one cell of each arm: the $G$ phase's class and `exact_remainder`, and every $T$ phase's, as
   §2.1 says. They must match the cell's own record.
4. **The wrapper.** Read it: it calls the original `phase_group_Levin_integral` and returns its
   result unchanged. Its transparency output shows `total` and `total_abserr` bit-identical,
   wrapped and unwrapped.
5. **Determinism and the readings.**
   - The determinism output is bit-identical in all four arms.
   - `state/readings.md` is later than the checks and earlier than the run's `manifest.json`.
   - It quotes prompt §4 word for word (check by substring).
6. **One cell, re-run by you.** Take one realistic cell of moderate cost, such as `T-first` at
   $10^4$ in arm P, through the script's own cell function, in a scratch process. Its `total`,
   `total_abserr`, `Levin_regions` and maximum depth must be bit-identical to its checkpoint record.
7. **The numbers, re-derived** from the checkpoint by a scratch reader:
   - the growth per shape and arm, and the level at 980;
   - the cells `not reached`;
   - S against P₀ and P₀ against P.

   They must match the memo. Then read the memo's §3 against prompt §4 as written. It names
   reading 1, reading 2 or neither, with its numbers, and does not invent a threshold.
8. **The representation term** is recorded for S, P₀ and P, with both declared errors, and with no
   verdict.
9. **The records.**
   - `log.md` follows campaign README §5.1, and every deviation is classified.
   - The memo has prompt §5's sections.
   - `issues.md` exists.

Write your verdict on each check to `var/handover-04/review.md`, with the commands you ran and what
they printed. End the turn with a report whose first line is the verdict (pass or stop). Then give:
- the growth per shape for each arm, and the level at 980;
- the cells `not reached`;
- which reading the numbers support, in the memo's words.

## 4. After the review — on the laptop

The laptop session is woken when this one ends. It copies `var/handover-04/` back with `rsync -a`,
without checkpoints or pickles, and re-checks check 6's cell record over `ssh macstudio-tunnel`
itself. Then, in **one records commit**, pushed:
- the scripts and outputs to `docs/handover/a4/`, scripts byte-identical and unformatted, with
  `docs/handover/a4/scripts/` added to `CLAUDE.md`'s `black` paragraph;
- the memo as `docs/handover/GK-PHASE-COST-GROWTH.md`;
- `log.md` to `logs/04-gk-phase-decision-test.md`;
- the board: A4's row and item;
- the narrowings of `[02-realistic-fixture-Gk-phase-is-the-superseded-construction]` and
  `[12-realistic-large-x-reads-the-retired-clamp-records]`;
- **`[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]`:** closed or narrowed **on the
  user's word**, after the laptop has put the memo's §3 to them;
- anything `issues.md` opens;
- `docs/OPEN_ISSUES.md` to match, with the count and date corrected by counting;
- this directory's README row for 04.

## 5. Stop and ask the user

Relay verbatim; do not adjudicate. The prompt's §8 conditions, or any failed check in §3: report
it, do not repair it. **Never write to a store or to another run's directory**, never move the
checkout, and never run a git command that writes.
