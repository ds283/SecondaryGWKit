# Orchestrator — prompt 04, remove the floor from seven rows

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code.** The arrangement is 03's ([`prompt-03.md`](prompt-03.md)), with the
differences §0 sets out.

**The prompt:** [`prompts/levin-phase-floor/04-remove-the-floor.md`](../04-remove-the-floor.md)
**Board item:** 04 · **Closes:** nothing · **Narrows:** `[00-consumer-anchoring-floor]`
(`GkTk-remedial`), `[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]` and
`[a3-baseline-unconverged-rows-are-stored-and-cannot-be-removed]` (`handover`),
`[03-m3-slope-clause-is-badly-posed-at-round-off]` (this campaign) · **Model:** Opus, as the prompt
recommends. You, the orchestrator, review on whatever model you were started with.

## 0. What makes this dispatch unusual

**You run on the Mac Studio, as 03's orchestrator did.** You are a Claude Code session over the
saved `MacStudio` SSH connection, in `/Users/ds283/Documents/Code/SecondaryGWKit`, the **main
checkout**. You dispatch the executing agent there, review there, and **commit nothing**. The
bookkeeping commit is made on the laptop (§4).

**The code runs in a second tree.** B1 (`049c7c8`) is now on the branch, and an A3 v2 row cannot be
re-run past it. So the prompt's code runs from a **git worktree at `98e68d3`**, which the laptop
has made, detached:

    /Users/ds283/Documents/Code/SecondaryGWKit-lpf04-98e68d3

- **The main checkout** is at the dispatch SHA, fast-forwarded by the laptop before starting you.
  It holds the prompt, 03's scripts under `docs/levin-phase-floor/03/scripts/`, the venv, the store,
  `var/` and the registry. The records go in its `var/levin-phase-floor-04/`.
- **The worktree** holds the code that runs. It has no venv and needs none: the agent calls the main
  checkout's `venv/bin/python` by its absolute path, with `PYTHONPATH` set to the worktree. The main
  checkout's `.git/info/exclude` lists `venv`, if a symlink is ever wanted, but none is.
- **No git command that writes**, in either tree, `fetch` included. The laptop moves both.
- **Prompt §4 step 1 is a check**, in both trees, not a move.

**You may share the host.** `handover` A3 (its prompt 03, refreshed in `f1a3888`) may be dispatched
to this main checkout while you run, at `HEAD`, read-only and Ray-free. It writes only under
`var/handover-03/`, and its own note forbids it to touch `var/levin-phase-floor-04/`, the worktree,
the store's files or any run directory. Neither of you uses Ray or the lock. Allow `var/handover-03/`
in check 1's scope, and nothing else of its.

**Facts checked by the laptop on 2026-10-25, 22:30 GMT,** over `ssh macstudio-tunnel`, read-only.
Pass them on.
- **The checkout:** `handover-remedial` at `8a02f0c` before the fast-forward, `git status --short`
  empty, one worktree (itself). Python 3.13.16. `98e68d3` is an ancestor of `HEAD`.
- **The registry:** nothing `running` or stale. The newest runs are `handover-12-r8-reference-…`
  and `levin-phase-floor-03-instrument-20261025T194950`, both `done`.
- **The lock is free:** `rehearsal_lock status` says so, `~/.raypoolkit/` is empty, and
  `~/ScienceData/.rehearsal-lock` does not exist. Note that `status` takes no `--project`.
- **03's run directory** holds `row_R1.pkl` to `row_R7.pkl`, `checkpoint.jsonl`, `manifest.json`
  and `status.json`.
- **`var/`** holds `datastores`, `handover-06`, `-07`, `-08`, `-10`, `-11`, `-12`,
  `levin-phase-floor-01`, `levin-phase-floor-03`, `py313-venv-02` and `runs`. There is **no
  `var/levin-phase-floor-04/`**. 115 GiB free; load about 4.5–7 on 10 cores, from system services,
  not from any run.
- **The store:** `store show var/datastores/handover-A3-v2-lambdacdm.sqlite` reports
  `problems: none` and the digest `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`.
- **What changed since `98e68d3`:** `git diff --stat 98e68d3 HEAD` touches `ComputeTargets/QuadSource.py`,
  `QuadSourceIntegral.py`, `TkSourceFunctions.py` and the new `handover.py` (B1), and nothing under
  `ObjectFactories/`, `config/`, `StoreFixtures/`, `main.py`, `runregistrykit.toml` or
  `requirements.txt`. So `store fingerprint` gives the same answer from either tree; run it from the
  main checkout.
- **The prompt's line references** are at `98e68d3`. `PrimitivePhase.theta_mod_2pi` is at
  `primitive_phase.py:285`, `build_phi_samples` `:359`, `CumulativeTable.delta` `:360`,
  `_composed_theta_mod_2pi` `:542`, `GkSourcePolicyData._build_phase` `:145`. There are two
  `PrimitivePhase(` constructions outside tests, $G$'s and $T$'s.
- **The run directory** will be `var/runs/levin-phase-floor-04-remove-<timestamp>/` in the main
  checkout. A run launched from a worktree is recorded there.
- **Shell.** Every command that takes arguments from a variable runs through `bash`, not zsh.

## 1. Before you dispatch

1. **The Gate**, re-read by you:
   - in the main checkout, `git rev-parse HEAD` is the dispatch SHA and `git status --short` is
     empty;
   - in the worktree, `git rev-parse HEAD` is `98e68d3` and `git status --short` is empty;
   - `./venv/bin/python -m runregistrykit list` shows nothing `running` or stale;
   - `./venv/bin/python -m raypoolkit.rehearsal_lock status`, a command of its own, says free;
   - `store show` reports `problems: none` and `0e32f24a…`;
   - 03's run directory holds its seven pickles;
   - there is no `var/levin-phase-floor-04/`.

   Anything else is a stop. Do not fetch.
2. **The `var/` marker:** `touch` a file in your session scratchpad immediately before dispatch,
   for check 1.
3. **No suite baselines.** The prompt changes no tracked file in either tree.

## 2. Dispatch

One fresh-context subagent, on **Opus**, in the main checkout. Give it the prompt, the dispatch
SHA, the worktree's path and §0 of this note. It does not read `orchestrator/`. Tell it plainly:
- **`HEAD` is not its own**, in either tree. Orchestration and other prompts' commits land on this
  branch.
- **Commit nothing, and run no git command that writes**, `fetch` included, in either tree.
- **Write only under the main checkout's `var/levin-phase-floor-04/` and its own run directory.**
  Nothing is written in the worktree. A write anywhere else is a stop: the store's files, its
  sidecar, 03's run directory and every other run directory included.
- **The predictions go in before the run.** Copy prompt §5 unchanged into `log.md`'s "Predictions
  and verdicts". Save the same text, with A1's list of covered groups, as `state/predictions.md`,
  **after step 5's gates pass and before step 7's launch**. Its mtime is check 6's evidence.
- **Save every command and its output.** The review reads the outputs, not the log's summary of
  them.
- **Launch, one check, end the turn.** Report the run id, the run directory, the detached Python
  pid, the `sh` pid if `nohup sh -c` is used, and the first checkpoint line. You wait on the pid
  once, in the background, and resume the agent with the run's final `status.json`. The agent does
  not wait or poll. The run may take up to about four hours.
- **Stop and ask** at any of prompt §7's conditions.

## 3. The review — on the Mac Studio, after the agent reports

1. **Scope.**
   - Both trees: `git status --short` is empty, and each `HEAD` is as in §1.
   - `find var -newer <marker>` lists only `var/levin-phase-floor-04/`, the run directory (with
     its launch log, if one was written beside it) and, if A3 ran, `var/handover-03/`. Not the
     store's files, not its sidecar, not 03's run directory.
   - `find <worktree> -newer <marker> -not -path '*/__pycache__/*'` lists nothing.
2. **The store.**
   - `diff` `state/store_files_before.txt` against `store_files_after.txt`: identical.
   - Re-run `store fingerprint` without `--write` yourself; it matches `0e32f24a…`.
   - There is no journal, `-wal` or `-shm` beside any store file.
   - The log says Ray was never initialised. `ray-processes --ray-temp-dir /tmp/ray-sgk`, run by
     you as a command of its own, lists nothing.
3. **The trees, read.** The scripts import this repository's packages from the worktree: check
   `PYTHONPATH` in the launch command and the run's manifest, and one module's `__file__` in a
   saved output if the agent printed it. Every output header records the script's SHA-256, the
   worktree's `HEAD` (`98e68d3`) and the main checkout's. The manifest records `98e68d3` with
   `git_dirty: false`.
4. **The arms, read.** In `scripts/phases.py`:
   - the patch is on `PrimitivePhase.theta_mod_2pi` at the class attribute, installed only around
     each unit's `evaluate_QuadSource_integral` call, and restored in a `finally`;
   - P calls the original method with its arguments unchanged and returns its value;
   - A changes $G$'s remainder only (by `sign`), with $z_{\rm top}$ a node of the `tau` table and
     $c_G$ from $\Delta^{\rm ext}$ in `mpmath`, at no fewer than 40 digits;
   - X's $\Delta^{\rm ext}$ sums the node limbs exactly and uses production's end partials;
   - every reduction is against `TWO_PI` taken exactly, into $(-\texttt{TWO\_PI}, 0]$, rounded once;
   - nothing else is patched: `raw_theta`, `theta_deriv`, the amplitudes, the partition, the driver
     and the tolerances.
5. **The gates.**
   - **E1:** `out/phases_04.txt` shows each bound of prompt §4 step 5 with its worst case, and none
     is exceeded.
   - **E2:** its medians under P equal 03's in `docs/levin-phase-floor/03/out/instrument_03.txt` to
     the digits printed there, and the bound holds under P, A and X. Under X each is below
     $10^{-2}F$.
   - **M0:** the comparison code uses `==` (or `is` for `None`) and the bit pattern, never
     `isclose`, on 03's fields. 7 of 7 rows are identical. **Independently:** for R1, read the
     stored `total` and `total_abserr` yourself (`sqlite3` with `mode=ro`) and compare them with the
     P checkpoint in `float.hex`.
6. **The predictions.**
   - `state/predictions.md`'s mtime is later than `out/phases_04.txt` and earlier than the run's
     `manifest.json`.
   - The log's "Predictions and verdicts" quotes prompt §5 word for word.
   - A1's covered groups are exactly the scored groups with $F^A_g \le 10^{-8}$ in
     `out/phases_04.txt`.
   - Each of R1, A1, N4 and R2 is scored holds, fails or undecided, against §5's rule as written,
     with its numbers. Q5's number is given for every row.
7. **The budget and the analysis.**
   - The run enforces the budget itself, at 3× each row's P wall time, and every over-budget unit is
     recorded as such and counted as §5 says.
   - `analyse.py` reads only the checkpoints. Re-run it into your scratchpad: it reproduces
     `out/remove_04.txt` apart from any timestamp.
   - Every group of every row is reported, scored or not. The depth histograms are split by type.
   - The descriptive items are present: $F^{\rm arm}_g$; $\sigma_\phi$ for every factor, production
     and rebuilt; the modulus term against $F$.
8. **The run.** `runregistrykit list` shows it `done`. Its manifest names `levin-phase-floor`, `04`
   and `remove`; `status.json` holds the pid. `stdout.log` and `stderr.log` are in the run
   directory.
9. **The records.**
   - `log.md` follows campaign README §5.1 and classifies every deviation. Step 1's check is one.
   - `verification.md` is written as `docs/levin-phase-floor/VERIFICATION.md` **§4 "The floor
     removed"**, with its links relative to `docs/levin-phase-floor/` in the `04/out/…` form.
   - `issues.md` exists, empty or not.

Write your verdict on each check to `var/levin-phase-floor-04/review.md`, with the commands you ran
and what they printed. End the turn with a report whose first line is the verdict:
- pass or stop;
- E1, E2 and M0;
- R1, A1, N4 and R2, with their verdicts and numbers, and Q5's number per row;
- the over-budget units, if any;
- the fingerprint.

## 4. After the review — on the laptop

The laptop session is woken when this one ends, and does the bookkeeping. It copies
`var/levin-phase-floor-04/` back with `rsync -a`, without the checkpoints (which stay in the run
directory) and without any CSV. It re-checks one thing over `ssh macstudio-tunnel` itself: check 5's
independent comparison on R1. Then, in **one commit**, pushed:
- everything to `docs/levin-phase-floor/04/`; the scripts byte-identical and unformatted, and their
  directory added to `CLAUDE.md`'s `black` paragraph if any output records their SHA-256;
- `verification.md` as `docs/levin-phase-floor/VERIFICATION.md` §4;
- `log.md` to `logs/04-remove-the-floor.md`;
- the board: 04's row, the hypotheses table, the gate G1 now open, and anything `issues.md` opens;
- the narrowings on the boards that hold them (`handover`, `GkTk-remedial`, and this campaign's
  `[03-m3-slope-clause-is-badly-posed-at-round-off]`);
- `docs/OPEN_ISSUES.md` to match, with the count and date corrected by counting;
- this directory's README row for 04.

The laptop also removes the worktree afterwards (`git worktree remove`) on the Mac Studio, once the
records are committed and the user agrees.

## 5. Stop and ask the user

Relay verbatim; do not adjudicate. Stop on the prompt's §7 conditions, or on any failed check in §3
here: report it, do not repair it. **Never write to the store**, and never run a git command that
writes.
