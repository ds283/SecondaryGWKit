## Prompt 06f of the levin-phase-floor campaign, on the Mac Studio — your brief

You are the one fresh-context agent running prompt 06f of the `levin-phase-floor` campaign of
SecondaryGWKit, on the Mac Studio (`mps026298.phys.susx.ac.uk`). An orchestrator session dispatched
you; it reviews your records afterwards. Today is 2026-10-31.

**Your prompt** is `prompts/levin-phase-floor/06f-cc-floor-terms-across-leaves.md`, read in the
worktree below. Read it in full, then everything its "Read first" list names, and the worktree's
`CLAUDE.md`, which binds you (the run registry, the venv, `black`). Read the campaign README
`prompts/levin-phase-floor/README.md` for §4.1 and §5 (§5.1 is the log's form). **Do not read
anything under `prompts/levin-phase-floor/orchestrator/`**, nor `docs/levin-phase-floor/06e/review/`,
`docs/levin-phase-floor/06e/review.md`, or anything under the main checkout's
`var/levin-phase-floor-06e/review/` or `var/levin-phase-floor-06f/review/`.

### Where things are

- **The dispatch SHA** is `f4770d13f86674381b3a0ef48b7ba05f8dda4934` (`f4770d13`). Its code is
  06e's code commit `f4d8e17377e7c82ca3a8096a03343d2abdf38a5c` (`f4d8e17`):
  `git diff --stat f4d8e17 f4770d13 -- . ':!docs' ':!prompts'` shows only `CLAUDE.md`
  (re-checked by the orchestrator today).
- **The worktree, your tree $g$:** `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06f-f4770d1`,
  detached at `f4770d13`, clean. It has no venv and no `var/`. Every capture and replay runs this
  tree's code. **Nothing is written in it**, no `__pycache__` included.
- **The main checkout:** `/Users/ds283/Documents/Code/SecondaryGWKit`, at `handover-remedial`
  `28ebaf0`, clean. It holds the venv, the stores, `var/` and the run registry. **No code runs from
  it, and it is never moved.**
- **Every command** calls `/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python` by that
  absolute path, with `PYTHONPATH` set to the worktree (plus your copied `scripts/` directory where
  the pickles need `harness`), and with `-B` or `PYTHONDONTWRITEBYTECODE=1`.
- **Your records** go in `/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06f/`:
  `log.md`, `verification.md`, `issues.md`, and `out/`, `scripts/`, `state/` (prompt §1, §5).
  The subdirectory `review/` there is the orchestrator's: never write in it.
- **The store** is the scratch copy of A3 v2,
  `/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06/wider/state/a3v2_copy/`,
  certified by `…/var/levin-phase-floor-06/wider/state/copy_files_certified.txt`. Open it read-only
  only; hash it before and after (prompt §1).
- **06e's census:** its records are in the worktree's `docs/levin-phase-floor/06e/census/` (and the
  same, plus the pickles, in the main checkout's `var/levin-phase-floor-06e/census/`); its run
  directory is `/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/levin-phase-floor-06e-census-20261030T081318/`,
  whose `checkpoint.jsonl` is the population (`docs/levin-phase-floor/06e/KEPT_ON_MAC_STUDIO.md`).
  Read them only.
- Read git only with `git --no-optional-locks`. **`HEAD` is not yours:** other sessions commit to
  this branch; check a tree's `HEAD` and `git status --short` where you rely on them.

### Facts (prompt 06f's orchestrator note §B.0, verbatim; check what you rely on — no fact here is to be trusted over the code or the checkpoint)

- **06e's census** `levin-phase-floor-06e-census-20261030T081318` is complete; its checkpoint is
  `a0dc4d7f…`, kept in full on the Mac Studio (`docs/levin-phase-floor/06e/KEPT_ON_MAC_STUDIO.md`).
- **The population exists in the checkpoint:** per call, `cc_floor` holds the summed terms
  (`node_term`, `abs_E`, `R_g`, `R_2`, `R_c`, `declared`, `t_value_rounding`, `t_argument`) over
  all and over floor-attributed Clenshaw–Curtis leaves. 06e's report counts node-dominated calls as
  491 and 1,323 and declared-dominated ones as 2,493 and 2,785 at `1e-7` and `1e-8`.
- **Z2's call** reproduces in about 105 s; its `cc_floor` has `declared` 0.0 and `node_term`
  $1.318\times10^{-33}$.
- **The tools exist:** `worker.py`'s `_cc_floor_terms` reproduces each leaf's terms with $E$ signed;
  `x1_capture.py` captures chosen calls and checks them bit for bit; the captures unpickle with the
  census `scripts/` on `PYTHONPATH` (the pickles name `harness` functions).
- **Production supplies `theta_abserr`** only through `BesselPhaseGroup.levin_theta`, so the
  declared term appears on analytic-block calls; the group calls (Z2's among them) carry none.
- **No Ray, no lock, no store built.** If anything needs Ray, that is a stop.

**Re-checked by the orchestrator on the Mac Studio at 20:27 UTC today, read-only:** the registry
holds 35 runs, all finished, none `running` or stale; both trees as above; the scratch copy's five
files match `copy_files_certified.txt` by SHA-256, with no journal, `-wal` or `-shm` beside them;
06e's census `checkpoint.jsonl` has SHA-256
`a0dc4d7f5c47eb28d8691b450559d005a76e1d9f3bd41442e249a419fb15904c`; the rehearsal lock is free and
`ray-processes --ray-temp-dir /tmp/ray-sgk` lists none; no `var/levin-phase-floor-06f*` existed
before the orchestrator's `review/`; 97 GB free.

### Rules

- **You write only** under the main checkout's `var/levin-phase-floor-06f/` (not its `review/`)
  and, for registered runs, `var/runs/`. **Never touch** `var/levin-phase-floor-07*` or any of prompt
  07's saved work or stashes; the scratch copy's directory beyond opening the copy read-only; any
  store (`var/datastores/` included: nothing is created there); any earlier prompt's records or runs,
  which you only read.
- **No git command that writes,** in any tree, `fetch` and `stash` included. You make no commit;
  the laptop copies your directory back and commits it.
- **No Ray.** Production modules import the `ray` package, and that is fine; a Ray session must
  never be initialised. **No rehearsal lock** is taken. If anything needs Ray, stop.
- **No code change** (prompt §0.3). Instrumentation is by wrapping module functions in your own
  process. If a measurement looks as if it needs a code change, stop.
- **At most six worker processes** at any time on this host.
- **Save every command you run and its output** under `out/` (or `state/` for launch transcripts).
  The review reads the outputs, not your log.
- **Predictions first.** Copy prompt §4's F1–F6 verbatim into `log.md` (and a timestamped file under
  `state/`) **before** any number is computed.
- **Long runs** (anything that may exceed about ten minutes) are registered with `runregistrykit`
  (`--campaign levin-phase-floor --prompt 06f`) and launched detached (`nohup sh -c …`, manifest
  first, stdout/stderr into the run directory, pid recorded). Verify in **one** check that it started,
  then **end your turn** and report the run id, run directory, detached Python pid, `sh` pid and the
  first heartbeat/checkpoint line. **Do not poll, do not `sleep`-wait, never `pgrep -f`.** The
  orchestrator waits on the pid and resumes you.
- **Every deviation from the prompt is numbered and classified** (`STRUCTURALLY REQUIRED`,
  `IMPLEMENTATION CHOICE` or `UNINTENDED DRIFT`) before you report, failed attempts included.
- **Stop and ask** at any of prompt §7's conditions, and at anything this brief or the prompt does not
  settle that changes what you measure. To stop, end your turn with the question stated plainly **as
  its first line**, followed by the state of every file and run you have made. Do not guess and do
  not repair.
- **Correctness is the only objective.** An argument from cost or time for a shortcut is a deviation
  to record, not a reason to take it.
- **Records:** `log.md` in campaign README §5.1's form, every heading of it (including **What
  shipped**); `verification.md` (the text of `docs/levin-phase-floor/VERIFICATION.md` §6f, with links
  in the `06f/…` form: the laptop commits your directory as `docs/levin-phase-floor/06f/`);
  `issues.md` (anything proposed for the board, or "none"); `out/results.md` (each verdict with its
  numbers, and for each term what a remedy would have to do — **no code proposed**); provenance in
  every output header; and a list, with SHA-256, of everything to stay on the Mac Studio (every pickle
  and every file over 2 MB).

When you are done, end your turn with a short report: F0 (and any replacements), F1–F6 each scored
with its key numbers, the deviations, and where the records are.
