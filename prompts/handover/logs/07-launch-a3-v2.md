# Log 07 — launch A3 v2 on the Mac Studio

> **Committed on the laptop** by the orchestrator (prompt 07 §6). A path `var/handover-07/…` below is
> that directory on the Mac Studio; its records are committed at `docs/handover/a3-v2/launch/`, and
> `verification.md` is `docs/handover/A3-V2-VERIFICATION.md` §2.


**Prompt:** prompts/handover/07-launch-a3-v2.md
**Commit:** none on the Mac Studio (prompt §6). Executed at dispatch SHA
`b90752dda46e57d588f8107cbd4d4557134fadb2`.
**Model:** Claude Opus 5.5, in the Claude Code session on the Mac Studio that ran 06.
**Date:** 2026-10-22 (21:05–21:13 BST).
**Result:** COMPLETE WITH DEVIATIONS. **A3 v2 is running**: run `handover-07-a3-v2-20261022T211155`,
driver pid 84436, with the lock's owner pid set to 84436.

The record is `verification.md` (§2, "The launch").

## What shipped

Nothing tracked. Under `var/handover-07/`:
- `out/prelaunch_checks_07.txt`, `out/lock_07.txt` and `out/launch_check_07.txt`;
- `launch_07.sh`, the launch command as run;
- `prelaunch.sh` and `launch_check.py`, throwaway helpers;
- `freeze_07.txt`;
- this log, `verification.md` and `issues.md`.

The launch itself made:
- `var/runs/handover-07-a3-v2-20261022T211155/` and `var/runs/handover-07-a3-v2.launch.log`;
- the store `var/datastores/handover-A3-v2-lambdacdm*`, being written;
- the lock `~/ScienceData/.rehearsal-lock`, held.

## Deviations from the prompt

1. **STRUCTURALLY REQUIRED — the launch needed the user's approval in this session.**
   - **What happened:** the first attempt (after `acquire` at 21:06:24) was denied by the session's
     permission classifier as a "production deploy". Nothing ran, which was checked afterwards.
   - **Done:** the agent stopped and asked. The user approved in chat ("I approve the launch:
     please go ahead"), and the same command ran at 21:11:54.
   - **Consequence:** the lock was held for about 5½ minutes with no pid and no build.
2. **IMPLEMENTATION CHOICE — the load check was re-sampled.**
   - **What happened:** §1.7's first `uptime` gave a one-minute load of 4.19, against §5's "loaded
     beyond 4 cores". The 5- and 15-minute averages were 3.69 and 3.62.
   - **Done:** the agent re-read it a minute later and got 3.82, then 3.75, with `top` showing
     67 %, then 84 %, of the CPU idle (about 1.6 of 10 cores busy, mostly `mediaanalysisd`). It
     launched on the re-sample.
   - **Alternative:** treat the single reading as a stop. It was not chosen, because the load
     average counts runnable threads rather than cores in use, and every other reading was under 4.
3. **IMPLEMENTATION CHOICE — the estimate, 39–44 h.** Prompt §0's 40–45 h was scaled by R4's QSI
   stage against R3's (88.7 s / 91.4 s = 0.970), since QSI dominates A3 v2. The whole-build ratio,
   0.953, would give 38–43 h. The figure is in the manifest's `purpose`.
4. **UNINTENDED DRIFT — a garbled header line in `prelaunch_checks_07.txt`.** The helper's §1.1 line
   used a `cut` with an empty range, which printed `cut: invalid decreasing range` into the record.
   It is cosmetic: the fetch and merge it describes are recorded in the session (3152af9 → b90752d,
   fast-forward), and every check below it ran. Left as written.
5. **IMPLEMENTATION CHOICE — §1.3 compared against two copies of 06's freeze**: 06's local
   `state/freeze_macstudio.txt` and the committed `docs/handover/a3-v2/rehearsal/out/pip_06.txt`.
   The first lookup of the committed file used a wrong path, so the comparison was re-run with the
   path `git ls-files` gives. Both diffs are empty.

## Verification performed

- §1.1–§1.6: all as stated (verification §2.1). §1.7 was recorded and found within limits on the
  re-sample.
- The Gate: `AutomaticallyInstallMacOSUpdates = 0`. `AutomaticDownload`, `CriticalUpdateInstall` and
  `ConfigDataInstall` are 1.
- `acquire` exit 0, the launch run by `bash`, then `set-pid 84436` exit 0, with `status` showing
  `pid: 84436`, alive.
- §3's one check passed on every point (verification §2.3): scope `full build:` with the pair,
  6 cpus, LambdaCDM and the 8-k grid; `git_dirty` false; `git_head` the dispatch SHA; the host is
  06's; the packages are at their pins; `script_sha256` is `b2fa4ef1…`; the state is `running` with
  `kill -0` ok; there are four `replaced` lines; `--zend 0.1`; no `--no-prune-unvalidated`; the
  local Ray bootstrap is there; `list` shows 1 alive.
- `git status --short` is empty after the launch and the records.

## Observations not acted on

1. **`CriticalUpdateInstall = 1` and `ConfigDataInstall = 1`** stay on, with `LastUpdatesAvailable =
   1`. These install security responses and data files, not macOS updates. Whether one can force a
   restart during the run is not established here. Not opened; for the user.
2. **`mediaanalysisd` used ~84 % CPU at launch.** It is background load on the 4 cores the build
   leaves free.

## State handed to the next prompt (08)

- **Run** `handover-07-a3-v2-20261022T211155`, driver pid **84436** (`sh` 84431), on host
  `d120d1cf03551e6f`, boot `C58A0F92-923F-471F-B32E-B1B9F5D0A848`. Launched 21:11:54 on 2026-10-22,
  with an estimated end around 2026-10-24 12:00–17:00 BST (39–44 h).
- **Store** `var/datastores/handover-A3-v2-lambdacdm.sqlite`, four shards, no sidecar yet.
- **Lock** held since 21:06:24, owner pid 84436. 08 releases it once the build's last Ray process
  is gone.
- **Checkout** at `b90752d`, clean, `venv` ignored by `.git/info/exclude:28`. Do not move it while
  the build runs.
- **To find it:** `./venv/bin/python -m runregistrykit list`. Liveness is `kill -0 84436` on the Mac
  Studio and the heartbeat, never a pattern match.
