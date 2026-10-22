# Prompt 07 — launch A3 v2 on the Mac Studio

**Campaign:** [`README.md`](README.md) · **Board item:** **07** (outside workstreams A–E) ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-22, with prompt 06, at `c081eed`. **Held** until 06 has landed and been
reviewed.
**Where it runs:** on the Mac Studio, in the Claude Code session 06 ran in or a fresh one in the
same folder. **It commits nothing there** (§6).
**Gate:**
- 06 landed and reviewed, its R4 stores both `39809dca…`;
- the dispatch commit is 06's records commit or later, on `origin/handover-remedial`, and **no
  commit after 06's dispatch SHA touches anything outside `prompts/` and `docs/`**:
  `git diff --stat <06's dispatch SHA> <dispatch SHA> -- . ':!prompts' ':!docs'` is empty;
- **the user has confirmed in chat** that automatic installation of macOS updates is off on the Mac
  Studio for the length of the run (a forced restart would hard-kill the build). It is a system
  setting; the agent does not change it.

**Opens:** anything you find, without fixing it.
**Recommended model:** **Opus**. One launch, and one check; the care is in the checks before it.

**Read first:**

1. `CLAUDE.md`, "Long-running jobs — the run registry" (all seven rules) and "The shared rehearsal
   lock".
2. [`06-a3-v2-host-and-rehearsal.md`](06-a3-v2-host-and-rehearsal.md) §0, and 06's log
   (`logs/06-a3-v2-host-and-rehearsal.md`), "State handed to the next prompt".
3. [`docs/a3-v2-readiness-verification.md`](../../docs/a3-v2-readiness-verification.md) §2.4, the
   launch checklist. §1 and §2 below are its first three items in current spelling.
4. [`docs/rrk-v020-adoption-verification.md`](../../docs/rrk-v020-adoption-verification.md) §3.5,
   what A3 v2 inherits, and `docs/handover/A3-V2-VERIFICATION.md` §1 (R4).

---

## 0. What this prompt is

**A3 v2** is the LambdaCDM store rebuilt from scratch at `QuadSourceIntegral`'s
`atol = 1e-32`, `rtol = 1e-7`, by the launcher's production scope: `main.py`'s own 9280-item
`QuadSourceIntegral` work list, 8 log-spaced $k$ over $10^5$–$3\times10^8$/Mpc, `--zend 0.1`, four
shards, 100 source samples per decade (`RunDrivers/quadsource_atol_sweep.py`: `K_MIN, K_MAX,
K_COUNT` at `:110`, `PRODUCTION_ZEND`, `_main_py_argv`). It is the store this campaign's A3 census
(prompt 03) and B2 read. The decisions behind it: `[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]`
(the pair), `[a3-baseline-quadsource-integrals-are-1680-short]` (regenerate from scratch), and 06
§0.2 (the host, `--cpus 6`).

**The estimate.** About 250 CPU-hours (`docs/handover/QUADSOURCE-TOLERANCE-SWEEP.md`, the
projection), so roughly 40–45 wall-hours at 6 CPUs at the laptop's per-core speed. Scale it by R4's
stage times against R3's, and record the scaled figure.

**This prompt launches it and stops.** Its turn ends after one check. What happens after the build
ends is prompt 08, held (§7).

---

## 1. The pre-launch checks

On the Mac Studio, in the checkout. Make `var/handover-07/out/` and save every command and its
output, in order, in `var/handover-07/out/prelaunch_checks_07.txt`.

1. `git fetch origin`, `git merge --ff-only origin/handover-remedial`; `git log -1 --format=%H` is
   the dispatch SHA; `git status --short` empty; the Gate's `git diff --stat` empty.
2. The launcher's SHA-256 equals 06's record of it.
3. `./venv/bin/pip freeze` equals 06's `pip_06.txt`.
4. `./venv/bin/python -m runregistrykit list`: R4's four runs, nothing `running` or stale.
5. `./venv/bin/python -m runregistrykit store show var/datastores/handover-A3-v2-lambdacdm.sqlite`:
   `absent`, no runs. `ls var/datastores/handover-A3-v2-lambdacdm*` finds nothing.
6. `PYTHONPATH=. ./venv/bin/python -m tools.rehearsal_lock status`: no lock. `ray-processes`:
   nothing. `echo "${RAY_ADDRESS-unset}"`: `unset`.
7. `uptime` (the load average), `df -h ~` (at least 50 GB free; the A3 baseline store was 339 MB,
   Ray's spill threshold is 95 % of the volume), `pmset -g` (`sleep 0`), and the ten processes
   using the most CPU.

Any of 1–6 not as stated is a stop. 7 is recorded; a full disk or a machine already loaded beyond
4 cores is a stop.

---

## 2. The launch

1. **The lock.** `PYTHONPATH=. ./venv/bin/python -m tools.rehearsal_lock acquire --purpose
   "SecondaryGWKit handover 07: A3 v2 build, about 40 h"`. Exit 3 means stop and ask the user.
2. **The launch**, verbatim but for `<SHA>`, the dispatch SHA's first seven characters, and
   `<hours>`, §0's scaled estimate:
   ```bash
   nohup sh -c 'PYTHONPATH=. ./venv/bin/python -u RunDrivers/quadsource_atol_sweep.py --build \
       --database var/datastores/handover-A3-v2-lambdacdm.sqlite \
       --job-name handover-A3-v2 --atol 1e-32 --rtol 1e-7 --cpus 6 \
       --register a3-v2 \
       --purpose "handover prompt 07: A3 v2, the LambdaCDM store rebuilt from scratch at QuadSourceIntegral atol 1e-32 rtol 1e-7 (main.py 9280-item work list, 8 k over 1e5-3e8/Mpc, zend 0.1, 6 cpus) on the Mac Studio at <SHA>; R4 (handover 06) reproduced 39809dca on this host; estimated <hours> h" \
       --campaign handover --prompt 07; \
     echo "** driver exit status $?"' \
     > var/runs/handover-07-a3-v2.launch.log 2>&1 < /dev/null &
   disown
   ```
   There is no `--rehearsal` and no `--resume`. The run id is `handover-07-a3-v2-<timestamp>`.
3. **`set-pid`** with the driver's pid, read from the run's `status.json`.

---

## 3. One check, then end the turn

Within a minute or two of the launch, **once**, and save it as `var/handover-07/out/launch_check_07.txt`:
- the manifest: `scope` begins `full build:` and names `atol=1e-32 rtol=1e-07`, `6 cpus`,
  `LambdaCDM`, `8 log-spaced k over 100000-3e+08 /Mpc` and the store; `git_dirty` `false`;
  `git_head` the dispatch SHA; `host` the Mac Studio's, equal to 06's; `packages` at the pins,
  none editable; `script_sha256` 06's;
- `status.json`: `state` `running`, its pid answering `kill -0`;
- `stdout.log`: four `replaced 1 occurrence` lines, the `main.py argv` line with `--zend 0.1` and
  without `--no-prune-unvalidated`, and the local Ray bootstrap;
- `./venv/bin/python -m runregistrykit list`.

Then **end your turn**. Do not wait, poll or start a background wait: the build runs for about two
days, and the registry, not this conversation, knows it exists (`CLAUDE.md` rule 4). The user
resumes the work when it has ended, by `list`.

If the check fails — the scope is the rehearsal's, the argv is wrong, the state is not `running` —
**stop and ask**. Do not signal the build.

---

## 4. If the build is interrupted

For the user and for 08; not for this prompt's turn.
- **A `SIGTERM`** (by a person, `kill -TERM <pid>`): the handler records `killed`/143 within about
  a second, then the fingerprint, and the launch log ends `** driver exit status 143`. Wait on the
  pid, not on `state`, then for each Ray process of this venv by pid
  (`[07-a-killed-builds-ray-agents-outlive-the-driver]`).
- **A hard kill** (power, OOM, a forced restart): `CLAUDE.md` rule 5, on the Mac Studio, including
  the `ray-processes` check after `end`
  (`[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]`).
- **The resume** is §2's command **exactly**, plus `--resume`, with `--register a3-v2-r<n>` and a
  launch log of that name: the same pair and no scope flag. `--resume` does not check the scope a
  store was built with. It prunes the unvalidated rows at open and recomputes only what is missing.
  The lock stays held, and `set-pid` is run again.

---

## 5. Stop conditions — stop and ask

- Any of §1.1–§1.6 fails, the disk has less than 50 GB free, or the machine is loaded beyond 4
  cores.
- The lock is held (exit 3), or a Ray process of this checkout is alive at `acquire` (exit 4).
- The launcher refuses (a `!!` line in the launch log): record it; do not retry with other flags.
- §3's check fails.
- The Gate's confirmation about macOS updates has not been given.

## 6. The records

**On the Mac Studio, under `var/handover-07/` only:** `out/` (§1, the lock's commands and exits as
`lock_07.txt`, §3); `log.md`, in `qcd-background-audit` README §5.1's form; `verification.md`,
**§2 "The launch"** of `docs/handover/A3-V2-VERIFICATION.md`; `issues.md` for anything opened.

**On the laptop, the orchestrator** copies them to `docs/handover/a3-v2/launch/`,
`logs/07-launch-a3-v2.md` and the verification document, updates this board (the 07 row and item,
§3 and §4) and `docs/OPEN_ISSUES.md`, and makes **one commit**. **The Mac Studio's checkout is not
moved while the build runs**: it stays at the dispatch SHA, clean, until 08.

## 7. What comes next: prompt 08, held

Written once the build has ended, because what it reads is what the build left. Its charter, fixed
now: `store create` and `store fingerprint --write` on the closed store; `docs/a3-v2-readiness-verification.md`
§2.4's after-build items (the census with verdict `no divergence`, `integrity_check` `ok` and
`foreign_key_check` empty on every file, `read_inventory` with no problem, one `version` row per
shard, no journal and no `replication_in_flight` row); every reader with `--ray-address local`,
leaving the store's files as they were and `store fingerprint` equal to the sidecar's; the
`QuadSourceIntegral` rows that reached `DEFAULT_LEVIN_MAX_DEPTH` or are stored unconverged, counted
against the A3 baseline's 2057 and 889; `release` of the lock; and where the archive copy of the
store and its run directories lives, which is the user's decision.
