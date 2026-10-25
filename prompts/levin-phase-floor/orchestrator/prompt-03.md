# Orchestrator — prompt 03, instrument seven depth-20 rows

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code.** The arrangement is 01's ([`prompt-01.md`](prompt-01.md)), with the
differences §0 sets out.

**The prompt:** [`prompts/levin-phase-floor/03-instrument-depth-20-rows.md`](../03-instrument-depth-20-rows.md)
**Board item:** 03 · **Closes:** nothing · **Narrows:**
`[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]` (`handover`),
`[00-consumer-anchoring-floor]` (`GkTk-remedial`), `[01-slow-group-leading-terms-do-not-cancel]`
(this campaign) · **Model:** Opus, as the prompt recommends. You, the orchestrator, review on
whatever model you were started with.

## 0. What makes this dispatch unusual

**You run on the Mac Studio, as 01's orchestrator did.** You are a Claude Code session over the
saved `MacStudio` SSH connection, in `/Users/ds283/Documents/Code/SecondaryGWKit`. You dispatch the
executing agent **in that checkout**, review there, and **commit nothing**. The bookkeeping commit
is made on the laptop (§4).

**You are alone in the checkout, and it must not move until you have ended.** `handover` 12 (B1)
was dispatched at the same time, but its code part runs on the laptop, in a worktree, and its
rehearsal R8 comes to the Mac Studio only after you have ended. B1 removes the hand-over clamp and
changes `QuadSourceIntegral`'s partition, so **this prompt's M0 can pass only at the pre-B1 tree**.
- **The laptop has already fast-forwarded the checkout** to the dispatch SHA, before starting you.
- **No git command that writes**, `fetch` included. A fetch is harmless in itself, but a merge
  after it could bring B1's code under a running harness.
- **Prompt §2 step 1 becomes a check:** `HEAD` is the dispatch SHA and `git status --short` is
  empty, with `pip freeze` recorded once. The agent records this as a `STRUCTURALLY REQUIRED`
  deviation: the laptop pins the tree.

**Facts checked by the laptop on 2026-10-25, 19:30 GMT,** over `ssh macstudio-tunnel`, read-only.
Pass them on.
- **The checkout:** `handover-remedial` at `757ef01` before the fast-forward, `git status --short`
  empty. Python 3.13.16.
- **The registry:** 10 runs, all finished; nothing `running` or stale.
- **The lock is free:** `~/.raypoolkit/` is empty, `status` says free, and
  `~/ScienceData/.rehearsal-lock` does not exist.
- **`var/`** holds `datastores`, `handover-06`, `-07`, `-08`, `-10`, `-11`,
  `levin-phase-floor-01`, `py313-venv-02` and `runs`. There is **no `var/levin-phase-floor-03/`**.
  117 GiB free; load about 3.5 on 10 cores.
- **The store:** `var/datastores/handover-A3-v2-lambdacdm.sqlite`, four shards
  `-shard000[0-3].sqlite`, and the sidecar `handover-A3-v2-lambdacdm.manifest.json`. `store show`
  reports `problems: none`, fingerprint `0e32f24a…`.
- **What changed since A3 v2's build.** `git diff --stat b90752d <dispatch SHA>` over
  `AdaptiveLevin`, `ComputeTargets/*.py`, `LiouvilleGreen`, `CosmologyModels`, `Quadrature`,
  `config` and `main.py` lists three files: `ComputeTargets/analytic_Gk.py` (prompt 09's domain
  guard), `config/ray_local.py`, and `main.py`, whose only change is `ray.init`'s `_temp_dir`. None
  reaches the Levin integration or the stage's set-up, so M0 is expected to hold. The prompt's
  remark that a difference in `analytic_rad` is recorded, not a stop, is the trace of the first.
- **The line references** of the prompt's "Read first" hold at the dispatch SHA:
  `DEFAULT_LEVIN_MAX_DEPTH = 20` at `levin_quadrature.py:154`, `_roundoff_floor` `:344`,
  `used_interval` `:450`, `_adaptive_levin` `:1694`, `adaptive_levin_sincos` `:2708` with the
  return documented by `:2836`; `QuadSourceIntegral.py:9` imports `adaptive_levin_sincos` and
  `:15` `build_phase_groups`, `evaluate_QuadSource_integral` is at `:734` and
  `phase_group_Levin_integral` at `:1010`; `build_QuadSourceIntegral_payload` at `main.py:358`,
  `largest_x_with_clearance` at `:1261`; the readers' `read_only=True` pool at
  `extract_QuadSourceIntegral_data.py:1318-1329`.
- **`StoreFixtures/`** holds `real_store_fixtures.py`, `shard_store_fixtures.py` and
  `standin_pool.py`. Whether `standin_pool` composes with `ShardedPool(read_only=True)` has not
  been tried; that is prompt step 3 (a)'s proof, and the agent's.
- **The rows** are in `docs/levin-phase-floor/01/out/candidates_01.txt`, at the dispatch SHA.
- **The run directory** is `var/runs/levin-phase-floor-03-instrument-<timestamp>/`, by the
  registry's naming. The pickles stay there.
- **Shell.** Every command that takes arguments from a variable runs through `bash`, not zsh.
- **If route (b) is taken:** each lock command is a shell command of its own, output by `>>`
  redirection, never a pipe; `--project SecondaryGWKit --ray-temp-dir /tmp/ray-sgk
  --ray-address in-process`, as `CLAUDE.md` gives it; `--purpose` names this prompt and the
  dispatch SHA, and no temp dir. `ray stop` is never run.

## 1. Before you dispatch

1. **The Gate**, re-read by you:
   - `git rev-parse HEAD` is the dispatch SHA, and `git status --short` is empty;
   - `./venv/bin/python -m runregistrykit list` shows nothing `running` or stale;
   - `./venv/bin/python -m raypoolkit.rehearsal_lock status` says free, as a command of its own;
   - `store show var/datastores/handover-A3-v2-lambdacdm.sqlite` reports `problems: none` and
     fingerprint `0e32f24a…`;
   - there is no `var/levin-phase-floor-03/`.

   Anything else is a stop. Do not fetch.
2. **The `var/` marker:** `touch` a file in your session scratchpad immediately before dispatch,
   for check 1.
3. **No suite baselines.** The prompt changes no tracked file.

## 2. Dispatch

One fresh-context subagent, on **Opus**, in the Mac Studio's checkout. Give it the prompt, the
dispatch SHA and §0 of this note. It does not read `orchestrator/`. Tell it plainly:
- **`HEAD` is not its own.** Orchestration and other prompts' commits land on this branch.
- **Commit nothing, and run no git command that writes**, `fetch` included.
- **Write only under `var/levin-phase-floor-03/` and its own run directory.** A temporary
  directory for step 3 (a)'s fixture proof is either under `var/levin-phase-floor-03/state/` or a
  `tempfile.TemporaryDirectory`, and the log says which. A write anywhere else is a stop, and that
  includes every file of the store, its sidecar and every other run directory. Route (b) adds the
  lock directory and `/tmp/ray-sgk`, and nothing else.
- **The predictions go in before the run.** Copy prompt §3 unchanged into `log.md`'s "Predictions
  and verdicts", and save the same text as `state/predictions.md` **before step 6's launch**. Its
  mtime is check 6's evidence.
- **Save every command and its output.** The review reads the outputs, not the log's summary of
  them.
- **Launch, one check, end the turn.** Report the run id, the run directory, the detached Python
  pid, the `sh` pid if `nohup sh -c` is used, and the first checkpoint line. You wait on the pid,
  once, in the background, and resume the agent with the run's final `status.json`. The agent does
  not wait or poll.
- **Stop and ask** at any of prompt §5's conditions. A stop with the lock held leaves it held and
  prints the owner file.

## 3. The review — on the Mac Studio, after the agent reports

1. **Scope.**
   - `git status --short` is empty, and `HEAD` is the dispatch SHA.
   - `find var -newer <marker>` lists only `var/levin-phase-floor-03/` and the run directory (and
     its launch log, if one was written beside it). Not the store's files, not its sidecar.
2. **The store.**
   - `diff` `state/store_files_before.txt` against `store_files_after.txt`: identical.
   - Re-run `store fingerprint` without `--write` yourself; it matches the sidecar's `0e32f24a…`.
   - There is no journal, `-wal` or `-shm` beside any store file.
3. **The route.** The log says which route was taken, and why.
   - **(a):** the fixture proof's transcript shows the fixture store opening read-only, an
     `object_get` returning a `QuadSourceIntegral`'s inputs, and the fixture's files byte-identical
     afterwards, all before A3 v2 was opened.
   - **(b):** the lock transcript: `acquire` 0, `set-pid` 0 with the harness's pid, `release` 0;
     `ray-processes --ray-temp-dir /tmp/ray-sgk`, run by you as a command of its own, lists
     nothing; `status` says free.
4. **The wrappers are transparent, read.** In the harness: each wrapper calls the wrapped function
   with the arguments it was given, unchanged, and returns its return value, the same object; it
   only records. Both module attributes are restored in a `finally`. The group label is parsed
   from `notify_label`, not inferred from call order.
5. **M0, checked.**
   - The comparison code uses `==` (or `is` for `None`), never `isclose` or `assertAlmostEqual`,
     on every field of prompt §2 step 5: `total`, `total_abserr`, `total_converged`, `WKB_Levin`,
     `numeric_quad`, `WKB_Levin_data.max_depth` and `num_regions`, and every group record of
     `metadata["WKB_Levin"]["subintervals"]`.
   - Its output reports seven of seven rows identical, field by field.
   - **Independently:** for each row, read the stored `total` and `total_abserr` from the store
     yourself (`sqlite3` with `mode=ro`, or the agent's reader, `-I`), and compare them with the
     pickled results. They must agree to the bit (`float.hex`).
   - `analytic_rad`'s comparison is reported, whichever way it went.
6. **The predictions.**
   - `state/predictions.md`'s mtime is earlier than the run's `manifest.json`.
   - The log's "Predictions and verdicts" quotes prompt §3 word for word.
   - Each of M1–M5 is scored holds, fails or undecided, with the numbers it was scored on, against
     §3's rule as written.
7. **The analysis, re-run.**
   - `analyse.py` reads only the pickles. Re-run it from them into your scratchpad: it reproduces
     `out/instrument_03.txt` apart from any timestamp.
   - **The definitions, read in `analyse.py`:** $E$ the group's `envelope` at the region's
     midpoint; $h$ the width in $\log(1+z')$; $D$ = `abserr`$/(E\,h)$; $\sigma_\theta$ from 65
     equispaced points, `numpy.unwrap`, a degree-4 fit in the local coordinate, RMS residual; the
     jitter-removed control a degree-12 fit to `theta_mod_2pi` on the depth-10 span, with nothing
     else changed; $c_s$ from the model's `wPerturbations`.
   - **The thresholds, read:** scored is at least 10 regions at depth 18 or more; M1 $[0.1, 10]$;
     M2 slope over depths 12–20 in $[-0.5, 0.5]$ and median $D/\sigma_\theta$ in $[0.01, 10]$, on
     six of seven rows; M3 slope at least 2 and a factor of at least 100, on six of seven; M4 share
     below 0.1; M5 within 5 % at all five abscissae on R1–R5 and R7.
   - The log says whether the ancestry discrepancy used the driver's own single-region routines or
     the Clenshaw–Curtis pair for every span.
   - Every group is reported, scored or not.
8. **The run.** `runregistrykit list` shows it `done`. Its manifest names `levin-phase-floor`, `03`,
   `instrument` and the launch pid, with `git_dirty: false` at the dispatch SHA. `stdout.log` and
   `stderr.log` are in the run directory.
9. **The records.**
   - `log.md` follows campaign README §5.1 and classifies every deviation. Step 1's no-op is one.
   - `verification.md` is written as `docs/levin-phase-floor/VERIFICATION.md` **§3 "Seven rows,
     instrumented"**, with its links relative to `docs/levin-phase-floor/` in the `03/out/…` form.
   - `issues.md` exists, empty or not.

Write your verdict on each check to `var/levin-phase-floor-03/review.md`, with the commands you ran
and what they printed. End the turn with a report whose first line is the verdict:
- pass or stop;
- M0, and M1–M5 with their verdicts and numbers;
- the verdict for the mechanism, and H-anchor and H-cancel group by group;
- the route taken;
- the fingerprint.

## 4. After the review — on the laptop

The laptop session is woken when this one ends, and does the bookkeeping. It copies
`var/levin-phase-floor-03/` back with `rsync -a`, without the pickles (which stay in the run
directory) and without any CSV (`.git/info/exclude` on the laptop excludes `*.csv`). It re-checks
one thing over `ssh macstudio-tunnel` itself: check 5's independent comparison on R1. Then, in
**one commit**, pushed:
- everything to `docs/levin-phase-floor/03/`; the scripts byte-identical, unformatted, and added to
  `CLAUDE.md`'s `black` paragraph if any output records their SHA-256;
- `verification.md` as `docs/levin-phase-floor/VERIFICATION.md` §3. If 02 has not landed, §3
  follows §1, and 02 inserts its §2 between them;
- `log.md` to `logs/03-instrument-depth-20-rows.md`;
- the board: 03's row, the hypotheses table, and anything `issues.md` opens;
- the narrowings on the boards that hold them (`handover`, `GkTk-remedial`, and this campaign's
  `[01-slow-group-leading-terms-do-not-cancel]`, which M5 decides);
- `docs/OPEN_ISSUES.md` to match, with the count and date corrected by counting;
- this directory's README row for 03.

**Whichever of 02, 03 and B1 lands later merges the others' board rows and index lines by reading
them, never by overwriting them.**

**Once this session has ended, the Mac Studio checkout is free for `handover` 12's R8.** The laptop
moves it, not you.

## 5. Stop and ask the user

Relay verbatim; do not adjudicate. Stop on the prompt's §5 conditions, or on any failed check in §3
here: report it, do not repair it. **Never write to the store**, and never run a git command that
writes.
