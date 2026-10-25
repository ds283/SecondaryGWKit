# Orchestrator — prompt 01, move the laptop's venv to Python 3.13

Read [`../README.md`](../README.md) first, all of it. Its conventions are `rpk-v030-adoption`'s
([`../../rpk-v030-adoption/orchestrator/prompt-01.md`](../../rpk-v030-adoption/orchestrator/prompt-01.md))
and, for the rehearsal, `rrk-v020-adoption`'s 03, carried over. This note says what differs.

**You do not write code.** You may:
- run the suites, read-only `pip` commands, `python -m runregistrykit list`, and the lock's `status`
  and `ray-processes`;
- wait on a driver's pid, or a Ray process's pid, in the background, once for each launch;
- run a probe from the session scratchpad, never committing one;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`01-move-the-laptop-venv-to-python-3.13.md`](../01-move-the-laptop-venv-to-python-3.13.md)
**Board items:** B1, P1, V1, G1, C1, S1, R6, K1 · **Closes:**
`[a3-v2-origin-main-moved-to-python-3.13-without-this-branch]` · **Opens:** nothing expected ·
**Model:** Opus, as the prompt recommends.

**Gate:** the prompt's, and `HEAD` is the commit that adds this note, or a later one that touches
no code.

## 0. What makes this prompt unusual

**The venv moves before the commit, and the commit does not undo it** (README §4.1). The review
checks the venv directly, not only the diff. If the agent stops after its §2.2, the venv is 3.13
and the 3.12 venv is in `~/Documents/Code/retired-venvs/`. The agent leaves it there, and moving
it back is the orchestrator's call or the user's.

**`git_dirty` is `true` at every launch.** The plan said `false`, which cannot hold. The pin edits,
`runregistrykit.toml` and `measure/out/` are uncommitted until the prompt's one commit, and
`git status --porcelain` counts them, as in `rpk-v030-adoption` 01 and `handover` 09 (R5). The
prompt was corrected before dispatch, in the commit that adds this note (its §2.10 and §4): the
stop is now a dirty `.py` file, not a dirty tree.

**The waits are the orchestrator's.** The prompt's §2.8 "waits" bullet has the agent wait in the
background. A subagent's turn is not resumed by its own background command, so the arrangement
is `rpk-v030-adoption` 01's:
- the agent launches detached, makes its one check, and **ends its turn**, reporting the driver's
  Python pid, the `sh` pid, the run id and the lock's owner file;
- the orchestrator waits on the pid once, in the background, and resumes the agent with the
  run's final `status.json`;
- for a `SIGTERM`ed or `SIGKILL`ed launch, the agent lists the Ray processes left and the
  orchestrator waits on their pids the same way;
- the agent never waits, polls or runs `pgrep -f`, and records the deviation as
  `STRUCTURALLY REQUIRED`.

**Lock commands: one per shell command, no pipe**, as `rpk-v030-adoption` 01's note §0 says. A
shell whose command line names `/tmp/ray-sgk` seeds the process rule, so its other children are
listed. Redirect with `>> file 2>&1`, record `$?` with the builtin `echo`, never use `| tee`, and
never put a lock command in the same shell command as a launch, a wait or the readers. No
`--purpose` names the temp dir.

**The hard-kill pair can end in a stop that is the user's.** At R3, Ray's GCS server outlived the
`SIGKILL` by 20 m 50 s and did not exit until the user ruled it be killed. If prompt §2.9's
ten-minute condition fires, the agent stops with the lock held. The orchestrator then reports the
pids, commands and parents to the user, who decides. **Nobody here kills it without that
ruling.**

**What moves on purpose:**
- `requirements.txt:75` and `:88`;
- the venv, replaced, with the old one moved aside;
- `runregistrykit.toml`'s `distributions` and the comment above it;
- one bullet in `CLAUDE.md`;
- `docs/py313-venv-verification.md`, which is new;
- `measure/out/01-*`, the log and the board;
- the `handover` board, and `docs/OPEN_ISSUES.md` for the closed issue;
- the `rrk-v020-adoption` board, if the GCS issue is narrowed;
- under `var/`: R6's six run directories and their launch logs, its three stores and their
  sidecars, and `var/py313-venv-01/`.

**What must not move:** every `.py` file, every test, every golden file, the three git packages
(same commits, not editable), every other pin, other campaigns' `prompts/` and dated `docs/`
(apart from the boards named above), and anything else under `var/`.

**Facts checked by the orchestrator on 2026-10-25 at `1fd1c36`.** Pass them on.
- The tree is clean, apart from the correction this note's commit carries. `git worktree list`
  shows the main checkout alone, equal to `origin/handover-remedial`.
- The registry holds 53 runs: 51 finished and 2 `unknown`, nothing `running` or stale.
- The venv is 3.12.15, with `ray==2.43.0` and `SQLAlchemy==2.0.39`.
  `/opt/local/bin/python3.13 --version` is 3.13.16.
- `~/Documents/Code/retired-venvs/` does not exist yet, so the agent's `mkdir -p` makes it.
- **The lock is free.** `status` exits 0 with `lock: free (… does not exist)`, `~/.raypoolkit/` is
  empty, and `~/ScienceData/.rehearsal-lock` does not exist. `ray-processes` exits 0 with `none`.
- **The rehearsal's inputs exist:**
  - R3's run directories `var/runs/rrk-v020-adoption-03-reference-20261018T232838` and
    `-resumed-c-20261018T233941`;
  - `docs/dk-and-rpk-migration/rehearsal/compare_with_recorded.py` and `run_readers.sh`;
  - the interrupter, `docs/datastore-integrity-verification/interrupt_when.py` (R3 ran it as
    `docs/rrk-v020-adoption-verification.md:830` shows).

  No `var/datastores/py313-venv-*` and no `var/py313-venv-01/` exists.
- **The planning measurements (README §0.2).** In a fresh 3.13.16 venv, a dry run of the two moved
  pins resolves 101 packages, and only the two pins change. `black --check --extend-exclude
  '^/prompts/' .` under 3.12 says 369 files would be left unchanged.
- The disk has 56 GB free. The load average is about 7 on 10 cores. README §7's four wall-clock
  tests may fire.

## 1. Before you dispatch

1. Check the Gate against §0's facts.
2. **The `var/` marker**, for check 7: `touch` a file in the scratchpad immediately before
   dispatch.
3. **No separate orchestrator baseline.** The agent's §2.1 under 3.12 is the baseline, taken
   before the venv moves. Check 4 re-runs the roots under 3.13. If a count looks wrong, the 3.12
   venv is still in `retired-venvs/` to re-measure with.

## 2. Dispatch

One fresh-context subagent, on **Opus**, in the main checkout. Give it the prompt's path, `HEAD`,
and §0 of this note up to "## 1. Before you dispatch". It does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the end**, staged by explicit path from the prompt's acceptance 6, then
  `git push`.
- **Baselines before the venv moves.**
- **`git_dirty` is `true`**, and a dirty `.py` file is the stop (§0).
- **Lock commands: one per shell command, no pipe** (§0).
- **The waits are the orchestrator's.** After each launch's one check, end the turn with the pids,
  the run id and the owner file (§0).
- **The wall-clock flakes** are re-run alone and recorded. They are not a stop.
- **Stop and ask** on any of the prompt's §4 conditions. If it stops with the lock held, it leaves
  it held and prints the owner file. It never moves the venv back.

## 3. The review — eight checks

1. **Scope.** `git show --stat <01>` lists exactly acceptance 6's paths.
   `git diff <01>~1 <01> -- requirements.txt` is the two lines, and `-- runregistrykit.toml` is
   `distributions` and its comment. No `.py` file is in the diff.
2. **The venv, measured here.** `./venv/bin/python --version` is 3.13.16, and `pyvenv.cfg` names
   `/opt/local/bin`. `package_provenance` of the seven distributions shows `ray` 2.59.0 and
   `SQLAlchemy` 2.0.55, with the three git packages at their commits and not editable.
   `pip check` is clean, and `pip install --dry-run -r requirements.txt` would install nothing.
   `pip freeze` diffed against `01-pip.txt`'s 3.12 freeze differs in the two lines only. The old
   venv is at the recorded path, its `pyvenv.cfg` 3.12.15.
3. **`CLAUDE.md` and the toml.** One new bullet in "Repository mechanics", in the file's voice, and
   the packages sentence unchanged. `distributions` lists the six names.
4. **The roots, re-run here at `<01>`**, serially. Each count equals the agent's §2.1 count, all
   OK, and `test_ray_address_default` passes. A known flake is re-run alone. `black --check` is
   clean.
5. **R6's four launches.** From `01-*.txt`:
   - `acquire` exited 0, and `set-pid` 0 at each launch;
   - during `reference`, `ray-processes` exited 1, listing `gcs_server` and `raylet` under
     `/tmp/ray-sgk`;
   - `reference` and `resumed-c` ended `done` with exit 0, and `resumed-a` and `resumed-b`
     ended `killed` with exit 143;
   - every manifest has `host.python` 3.13.16, the seven `packages` and `git_head` the dispatch
     SHA, and is dirty in no `.py` file;
   - **both stores are `39809dca…`.** Recompute them here with `store fingerprint` (no
     `--write`), and confirm `compare_with_recorded.py` was empty against R3;
   - the readers exited 0, with the store unchanged.
6. **The hard-kill pair.** `hardkill-a` was `SIGKILL`ed (exit 137) and went `stale`. The resume was
   refused with no run directory made. `end` exited 0, `lost` and verified, and a second `end` was
   refused. `hardkill-b` ended `done` with its store `39809dca…`. Record the GCS server's lifetime
   after the kill against R3's 20 m 50 s. If it narrowed the GCS issue, check the board entry and
   the index row.
7. **Nothing else moved.** `find var -newer <marker>` lists only §0's paths. `git status --short`
   is clean. The lock is free, and `ray-processes` lists nothing.
8. **The records.**
   - The log has every README §5.1 heading and classifies each deviation, including the waits and
     `git_dirty`.
   - `docs/py313-venv-verification.md` §1 opens with the venv.
   - The `handover` board's entry has moved to §4 with the closing line, and this board's §4
     records it.
   - `docs/OPEN_ISSUES.md` has lost the row from §1.23, and its count is 102.
   - The commit message is in `CLAUDE.md`'s form, ending `Co-Authored-By: Claude Opus …`.

## 4. After the review

If it passes, fix the residue first, in a follow-up commit of its own:
- "this commit" becomes the SHA, on the board and in the log;
- README §2's 01 row status;
- `prompts/INDEX.md`'s row ("1 / 2 landed").
Then record the review as `### 5.2 Review of 01 (<date>)` on the board, and **release the hold on
02**, whose orchestrator note is written next.

**Report to the user:**
- the venv and its two moved pins;
- the roots;
- R6's fingerprint;
- what the hard-kill pair showed about Ray 2.59.0's GCS server;
- any new log lines;
- that R6's three stores are theirs to retire (README §3);
- that the 3.12 venv in `retired-venvs/` is theirs to delete once 02 has landed.
