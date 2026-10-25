# Orchestrator — prompt 02, move the Mac Studio's venv to Python 3.13

Read [`../README.md`](../README.md) first, then [`prompt-01.md`](prompt-01.md), whose arrangement
this note carries over to the Mac Studio, then 01's review on the board (§5.2). This note says
what differs.

**The prompt:** [`02-move-the-mac-studio-venv-to-python-3.13.md`](../02-move-the-mac-studio-venv-to-python-3.13.md),
amended before dispatch in the commit that adds this note (Gate, steps 4, 5 and 8, acceptance 5a)
· **Board items:** M1, M2, R7, A1 · **Closes:**
`[01-ray-2.59-enables-token-auth-and-writes-a-token-into-home]`, when acceptance 5a holds ·
**Model:** Opus, as the prompt recommends.

## 0. Where this runs, and who does what

**The orchestrator runs on the Mac Studio**, as `handover` 10's did: a Claude Code session over the
saved `MacStudio` SSH connection, in `/Users/ds283/Documents/Code/SecondaryGWKit`. The laptop
session that wrote this note starts it.

**What you do there.** You do not write code.
- Check the Gate and dispatch **one fresh-context subagent, on Opus, in that checkout** to execute
  the prompt.
- Do the waits.
- Review the agent's records there, where the stores and the readers' output are.
- **Commit nothing.**

**What the laptop does afterwards.** The bookkeeping commit is made on the laptop, as the prompt's
§6 says: the records copied, the board, the index and the verification record's §2.

**When you are done**, end your turn with a report in the form of §4. The laptop session gets its
first lines, and reads the rest from your `var/py313-venv-02/` records over ssh.

**You may:**
- run the suites, read-only `pip` commands, `runregistrykit list`, `store show` and
  `store fingerprint` without `--write`, and the lock's `status` and `ray-processes`;
- wait on a driver's pid, or a Ray process's pid, in the background, once for each launch;
- run a probe from your session scratchpad, never writing in the checkout outside `var/`;
- after the review passes, retire R7's two stores (§3).

## 1. The choices carried over from 01, and the user's since

- **The waits are yours.** After each launch the agent makes its one check and ends its turn. It
  reports:
  - the driver's Python pid, the `sh` pid, the run id, the interrupter's pid if one runs, and the
    lock's owner file.

  You wait on the pid once, in the background, and resume the agent with the run's final
  `status.json`. After a `SIGTERM`ed launch the agent lists any Ray processes left, and you wait
  on those pids too. The agent never waits, polls or runs `pgrep -f`, and it records the
  arrangement as a `STRUCTURALLY REQUIRED` deviation.
- **Lock commands: one per shell command, no pipe.**
  - Redirect with `>> file 2>&1`, and record `$?` with the builtin `echo`. Never use `| tee`.
  - Never put a lock command in the same shell command as a launch, a wait or the readers.
  - No `--purpose` names the temp dir.
  - **Your own shells too.** A shell command whose text names `/tmp/ray-sgk` is itself listed by
    `ray-processes`. The laptop's ssh check on 2026-10-25 counted itself this way ("1"). Check for
    Ray by name (`ps -Ao pid,ppid,command | grep -E "gcs_server|raylet"`) when in doubt.
- **`git_dirty` is `false` here.** The prompt writes nothing tracked, and `var/` is ignored. A
  `true` is the stop the prompt names.
- **The hard-kill pair is not run** (README §6.2 C4). If a `SIGTERM`ed launch leaves a Ray
  process alive ten minutes after its driver's death, stop with the lock held and report it. The
  user decides, as in 01, and nobody signals it without that ruling.
- **Token authentication, the user's decision of 2026-10-25.** `RAY_AUTH_MODE=token` is set by
  `config/ray_local.py` at the dispatch SHA (`d833ee7`). The token is generated and made mode 600
  **before any Ray start** (prompt step 4). R7 is the first real-Ray run of the change, because the
  laptop's smoke test was refused by a lock StochasticInstantons held. Its four checks (step 8)
  decide acceptance 5a.
  - **The token never reaches a saved output.** `get-auth-token` prints it, so its stdout goes to
    `/dev/null`. Never `cat` the file.
- **The throwaway stores are retired after the review**, on the user's decision for R6's (§3).

**What moves on purpose, all under the Mac Studio's checkout or `~`:**
- the checkout, fast-forwarded;
- the venv, replaced, with the old one moved to `~/Documents/Code/retired-venvs/`;
- `~/.ray/auth_token`, new, at mode 600;
- under `var/`: R7's four run directories, their launch logs, two stores with sidecars (retired
  after the review), and `var/py313-venv-02/`.

**What must not move:** any tracked file; A3 v2's five files and its sidecar; every other store and
run under `var/`; `~/Documents/Code/retired-venvs/SecondaryGWKit-py313-20261022`; anything else
on the Mac Studio.

**Facts checked by the laptop on 2026-10-25, read-only over `ssh macstudio-tunnel`.** Pass them on.
- The checkout is on `handover-remedial` at `57f67a9`, clean, and behind
  `origin/handover-remedial`. The fast-forward will carry 01, its review, `28f94c7`, `d833ee7` and
  this note.
- The venv is 3.12.15, and `/opt/local/bin/python3.13` is present (3.13.16 on 2026-10-25,
  morning).
- `runregistrykit list` shows five runs, all finished. The lock is free.
- No Ray process is running. `ps` by name finds no `gcs_server`, `raylet` or worker.
- `~/.ray/` holds only `config.json`, so there is no `auth_token`. `retired-venvs/` holds only
  `SecondaryGWKit-py313-20261022`.
- The load is about 3 on 10 cores. A3 v2 is in `var/datastores/` as `handover` 10 left it.

## 2. Before you dispatch

1. Check the Gate as amended. The `git diff --stat b90752d <dispatch SHA> -- ObjectFactories
   config main.py` must list `config/ray_local.py`, `main.py` and
   `ObjectFactories/tests/test_quadsource_parent_key.py`, and nothing else.
2. The `var/` marker: `touch` a file in your scratchpad immediately before dispatch.
3. Pass the agent the prompt's path, the dispatch SHA, and §1 of this note. The agent does not read
   `orchestrator/`.

## 3. The review, and after it

**The review**, on the Mac Studio, from the agent's records and by measuring directly:
1. **Scope.** `git status --short` is empty and `HEAD` is the dispatch SHA. `find var -newer <marker>`
   lists only §1's paths. A3 v2's five files have the SHA-256, size and mtime of step 6's
   `state/store_files_before.txt`.
2. **The venv.** It is Python 3.13.16. `pip freeze` diffed against the laptop's (the last block of
   `prompts/py313-venv/measure/out/01-pip.txt`) is empty, and `pip check` is clean. The 3.12 venv
   is in `retired-venvs/`.
3. **The roots.** Re-run here serially: 01's counts, with RunDrivers at 124. The known wall-clock
   flake is re-run alone.
4. **R7.** `reference` and `resumed-c` ended `done`, and `resumed-a` and `resumed-b` `killed`.
   `store fingerprint` without `--write` gives `39809dca…` for both stores, and
   `compare_with_recorded.py` against R3's two run directories (`var/handover-06/recorded/`) is
   empty. Every manifest has `git_dirty: false`, `host.python` 3.13.16 and the seven distributions.
5. **Token mode.** The prompt's four checks of step 8, re-made here. Grep every R7 log and every
   reader log for `Token authentication is enabled` and for `Generated new authentication token`:
   both give none. `ls -l ~/.ray/auth_token` is mode 600.
6. **A3 v2.** `store fingerprint` without `--write` gives `0e32f24a…`, the files are unchanged, and
   every reader exited 0.
7. **The lock** is free, and `ray-processes` lists nothing.
8. **The records** under `var/py313-venv-02/`: `out/`, `state/`, `log.md` (README §5.1's headings,
   each deviation classified), `verification.md` (§2 of `docs/py313-venv-verification.md`, in §1's
   shape) and `issues.md`.

**After the review passes:** retire R7's two stores. For each:
- `store retire --dry-run --reason "py313-venv 02: R7's throwaway <slug> store, 39809dca, retired after 02's review on the user's decision (2026-10-25)" var/datastores/py313-venv-02-<slug>.sqlite`.
  Check that the fresh fingerprint matches, only R7's own runs name the store, and the five files
  are listed;
- then the same without `--dry-run`, and `store show`, which says `retired`.

Record all of it in `var/py313-venv-02/out/retire_02.txt`. **Never retire A3 v2** or any store not
named `py313-venv-02-*`. The readers' output, `var/py313-venv-02/extract/`, stays: it is not a
store, and the user decides it.

## 4. Report

End your turn with a report whose **first line** is the verdict, such as "py313-venv 02: review
passed, R7 39809dca, A3 v2 0e32f24a, token mode shown". Then give:
- the dispatch SHA;
- the freeze comparison and the roots' counts;
- R7's fingerprints, and A3 v2's under 3.13;
- the four token-mode checks;
- any new log line, and every deviation with its classification;
- the retirements;
- what is left for the laptop's commit, which is the prompt's §6.
