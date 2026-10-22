# Orchestrator — prompt 07, launch A3 v2

Read [`README.md`](README.md) in this directory first, and [`prompt-06.md`](prompt-06.md), whose
arrangement this prompt reuses. **You do not write code.**

**The prompt:** [`prompts/handover/07-launch-a3-v2.md`](../07-launch-a3-v2.md)
**Board item:** 07.

## 0. What makes this prompt unusual

It starts the build this campaign exists to make, about two days of the Mac Studio, and then
stops. Everything that can go wrong goes wrong before the launch or in the first minute. Your
review turns on the launch check (prompt §3) and on the session having **ended its turn** after it,
not waiting.

## 1. Before you dispatch

1. **06 has landed and been reviewed**: `319b14f`, R4 `39809dca…` on the Mac Studio.
2. **The Gate's confirmation.** The user has said in chat that automatic macOS update installation
   is off on the Mac Studio. Do not dispatch on anything less. As found on 2026-10-22,
   `defaults read /Library/Preferences/com.apple.SoftwareUpdate AutomaticallyInstallMacOSUpdates`
   gave `1`. Read it again (`ssh macstudio-tunnel`) and record what it gives. A `1` after the user's
   confirmation is reported to the user, not overridden.
3. **The dispatch SHA** is the commit carrying this note, pushed to `origin/handover-remedial`.
   `git diff --stat 3152af9 <dispatch SHA> -- . ':!prompts' ':!docs'` is empty: R4's code is the
   code A3 v2 builds.
4. **The laptop**: nothing long running (`./venv/bin/python -m runregistrykit list`).

## 2. Dispatch

To the Mac Studio session of 06 (`handover-06-r4`) if it is still open, since it has the host's
context, or else a fresh session there in the same folder. Give it the prompt, the dispatch SHA and
these instructions, besides the prompt's own:
- **Run every command that takes arguments from a variable through `bash`, not zsh.** 06 lost the
  lock's pid for `reference` and a hash comparison to zsh, which does not split a word in an unquoted
  variable and reads a bare `==` as `=command` expansion (06's log, deviations 8–10). Here the
  launch and `set-pid` are the commands that matter.
- **`.git/info/exclude` carries `venv`** (06, deviation 2). Check `git check-ignore -v venv` before
  the launch; without it the manifest records `git_dirty: true`.
- Commit nothing; write only under `var/handover-07/`, plus the lock and the run the launch makes.
- **End the turn after prompt §3's one check.** No background wait, no poll.

## 3. The review — on the laptop, after copying `var/handover-07/` back

1. **The pre-launch checks** are all as prompt §1 says, the macOS setting is among them, and the
   checkout's `HEAD` is the dispatch SHA with an empty `git status --short`.
2. **The launch command** in the log is prompt §2's verbatim, with only `<SHA>` and `<hours>` filled.
3. **The manifest**, read from the Mac Studio yourself (`ssh macstudio-tunnel`, read-only): `scope`
   begins `full build:` with `8 log-spaced k over 100000-3e+08 /Mpc`, `atol=1e-32 rtol=1e-07`,
   `6 cpus`; `git_dirty` `false`; `git_head` the dispatch SHA; `host.id` `d120d1cf03551e6f`;
   `script_sha256` `b2fa4ef1…`; the three packages as at R4.
4. **The status** is `running`, its pid answers `kill -0` on the Mac Studio, and the lock's owner
   file names that pid.
5. **`stdout.log`**: four `replaced 1 occurrence` lines, `--zend 0.1`, no `--no-prune-unvalidated`,
   the local Ray bootstrap.
6. **Bookkeeping**: `out/` to `docs/handover/a3-v2/launch/`, the log to `logs/07-launch-a3-v2.md`,
   `verification.md` as §2 of `docs/handover/A3-V2-VERIFICATION.md`; the board's 07 rows; §3, §4
   and `docs/OPEN_ISSUES.md` for anything opened. One commit, then push. **The Mac Studio's checkout
   is not touched**: it stays at the dispatch SHA until 08.

## 4. Stop and ask the user

Relay verbatim; do not adjudicate. The prompt's §5; any failed check in §3. **Never signal the
build** on a review finding: report it, and the user decides.
