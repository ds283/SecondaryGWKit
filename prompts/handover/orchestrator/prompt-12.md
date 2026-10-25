# Orchestrator — prompt 12, B1: remove the $T_k$ seam gap, by overlap

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code**, and the deliberate breakages are the agent's, not yours.

**The prompt:** [`prompts/handover/12-remove-the-tk-gap.md`](../12-remove-the-tk-gap.md)
**Board item:** B1 · **Closes, when R8 holds:** `[08-handover-clamp-error]`,
`[12-handover-clamp-error-in-production]`, `[05-numeric-region-is-now-the-accuracy-floor]` as far
as the $T_k$ hand-over goes (all `source-remediation`) · **Narrows:**
`[06-source-spline-residual-vs-handover]` · **Model:** Opus, as the prompt recommends, for both
parts.

The prompt runs in two parts, and so does this note:
- **Part A (§1–§5 of the prompt), the code,** on the laptop: a fresh-context subagent, one commit.
  This note's §A.
- **Part B (§6), rehearsal R8,** on the Mac Studio, orchestrated there as `handover` 10 and 11
  were. This note's §B.

## 0. The order, and why

Dispatched together with `levin-phase-floor` 03 on 2026-10-25 at the user's request. Three things
fix what may overlap.
1. **03 needs the pre-B1 tree.** Its M0 re-runs seven A3 v2 rows and must reproduce the stored
   values bit for bit, through the clamp and the partition B1 removes. So **the Mac Studio's
   checkout stays at the dispatch SHA until 03's orchestrator has ended**, and R8 cannot start
   before then.
2. **`levin-phase-floor` 02 is running on the laptop, in the main checkout.** Its registered
   probe, `levin-phase-floor-02-noise-probe-20261025T183809`, pid 70545, was alive at 19:25 GMT,
   47 minutes into one cell at the depth cap. Its heartbeat is older than 900 s, so `list` shows it
   stale, but its pid answers `kill -0`. It is another session's, uses no Ray and no store, and is
   not ours to end. Its script, `docs/levin-phase-floor/noise_injection.py`, imports
   `ComputeTargets/tests/test_quadsource_integral.py` and `docs/handover/realistic_large_x.py` from
   that checkout. B1 edits the first, and removes the `clamp_gaps_log1pz` key the second reads
   (`:332`, `:666`). So **Part A works in a worktree of its own**, and **its commit lands on
   `handover-remedial` only after 02 has landed and no 02 run is alive.**
3. **Part A needs no Ray and no store**, so it can run while 03 runs on the Mac Studio.

So: 03 on the Mac Studio and Part A on the laptop now; then Part A lands, after 02; then R8, after
03's orchestrator has ended.

---

## A. Part A — the code, on the laptop

### A.0 Facts checked by the orchestrator at the dispatch SHA

Pass them on.
- **The Gate as amended.** The registry shows nothing `running` or stale except 02's probe (§0
  item 2), whose pid is alive. Part A launches no run, uses no Ray and opens no store, so the
  agent records the exception as a `STRUCTURALLY REQUIRED` deviation, not a stop. **Any other**
  `running` or stale run is a stop.
- **The worktree.** `/Users/ds283/Documents/Code/SecondaryGWKit-handover-12`, on the branch
  `handover-12-b1` cut at the dispatch SHA, with `./venv` a symlink to the main checkout's (the
  common `info/exclude` already lists `venv`). **The agent works only there.** It never touches the
  main checkout, which holds 02's untracked script and its running probe. A run directory or store
  under the worktree's `var/` is not expected; the agent writes none.
- **The Gate's grep,** outside `docs/`, `prompts/` and `venv/`, finds two files:
  `ComputeTargets/QuadSourceIntegral.py` (`:111`, `:196`, `:243`, `:301`, `:310`, `:348`, `:438`,
  `:509`, `:548`, `:800`, `:1036`) and `ComputeTargets/tests/test_quadsource_integral.py` (`:47`,
  `:51`, `:847-848`, `:891-892`, `:943`, `:950`). **No `extract_*` reader reads a removed key**, so
  prompt §3.3's reader adjustment is empty; the agent confirms it.
- **The grep's `docs/` hits,** which the agent records and does not edit:
  - `docs/handover/realistic_large_x.py` reads `item["clamp_gaps_log1pz"]` at `:332` and `:666`.
    After B1 those lines raise `KeyError`. The prompt forbids editing the file (its checkpoint is
    gated on its own SHA-256), and says a re-run after B1 is B2's. Record it under "Observations
    not acted on" and open it as a §3 issue for B2.
  - `docs/levin-phase-floor/noise_injection.py` (02's, untracked in the main checkout at
    dispatch) imports that module and `test_quadsource_integral`'s `Case`. After B1 it is a record
    of its own SHA, not a script that runs at `HEAD`. Record it beside the above.
  - `docs/handover/a3-v2/seam/scripts/seam_census.py` is a record, byte-identical to its run.
  - `docs/source-remediation-verification/analyse_quadsource_integral.py:297` reads the key with
    `.get`, so it reads a post-B1 row as gap-free.
- **`numeric_crossover_z`** is used outside `QuadSource.py` (`:31`, `:128-129`, `:391`) only by
  `ComputeTargets/tests/test_quadsource.py` (`:26`, `:250-259`). Prompt §2.1's alias question is
  answered by that test.
- **The line references** hold: the region walk and the "audit A3" comment at
  `QuadSource.py:110-150` and `:126`, `_lookup` at `:161`; the crossover cross-check at
  `QuadSourceIntegral.py:782-792`; `_check_region_covers` at `:573`; `Fixture` at
  `test_tk_source_functions.py:197`, `drop_first_WKB_sample=False` at `:215`, the
  `stop_deltaz_subh` cross-check test at `:906-911`.
- **The roots in a worktree.** The baselines below were taken in the worktree, so any test that
  reads a gitignored file under the main checkout's `var/` behaves the same way in both counts.
- **The known flake** (`test_tk_wkb_phase … test_wall_time_per_object`) is re-run alone and
  recorded. It is not a stop. The laptop carries 02's probe on one core, and a ChronoSync backup
  may load it further; record `uptime` beside the roots.

### A.1 Before you dispatch

1. Commit this note, and push; that commit is the dispatch SHA.
2. `git worktree add -b handover-12-b1 ../SecondaryGWKit-handover-12 <dispatch SHA>`; symlink
   the venv; `git -C ../SecondaryGWKit-handover-12 status --short` is empty.
3. **Baselines**, the seven roots serially in the worktree, into the session scratchpad.
4. **The marker:** `touch` a file in the scratchpad immediately before dispatch.

### A.2 Dispatch

One fresh-context subagent, on **Opus**. Give it the prompt, the dispatch SHA, the worktree's
path, the baselines and §0 and §A.0 of this note. It does not read `orchestrator/`. Tell it
plainly:
- **Work only in the worktree.** Every command runs there with `PYTHONPATH=.`. The main checkout
  is not touched, read or written; a write there is a stop.
- **One commit, at the end, on `handover-12-b1`**, staged by explicit path. It covers
  `ComputeTargets/TkSourceFunctions.py`, `QuadSource.py`, `QuadSourceIntegral.py`, a helper module
  under `ComputeTargets/` if prompt §2.1's helper gets one, the test modules under
  `ComputeTargets/tests/`, `docs/handover/HANDOVER-B1.md`, `logs/12-remove-the-tk-gap.md`, the
  `handover` board, and `docs/OPEN_ISSUES.md` if the commit opens an issue. A path outside that
  list is a stop. **No push, no merge, no rebase:** landing is the orchestrator's (§A.4).
- **The header's issues are not closed in this commit.** They move to §4 only in the records
  commit after R8 (prompt §5).
- **Stop and ask** at any of prompt §8's conditions.

### A.3 The review — eight checks, in the worktree

1. **Scope.** `git show --stat handover-12-b1`: only §A.2's paths.
   `docs/handover/realistic_large_x.py` and every `TkNumericIntegration`, `TkWKBIntegration`,
   `numeric_with_phase_cut`, factory, schema and grid file are byte-identical to the dispatch SHA
   (`git diff --stat <dispatch SHA> handover-12-b1 -- ObjectFactories Quadrature main.py
   docs/handover/realistic_large_x.py ComputeTargets/TkNumericIntegration.py
   ComputeTargets/TkWKBIntegration.py` is empty).
2. **The three definitions, read.** `z_init` is the extremum, still cross-checked against
   `z_exit - stop_deltaz_subh`. `crossover_z` is the switch node, matched to the numeric values by
   `store_id` or exact float equality, never by a $\log(1+z)$ round trip. `HANDOVER_OVERLAP_NODES
   = 2`, named once. `handover_switch_node` is defined in one module and called by both consumers.
3. **The clamp is gone.** The Gate's grep, re-run at `handover-12-b1`: no hit outside `docs/` and
   `prompts/`, or each remaining one explained in the log. Every factor and the source get
   `_check_region_covers`, with no tolerance.
4. **The roots, re-run here.** Six equal to the baselines; `ComputeTargets` up by exactly the test
   methods added, counted from the diff. All OK, with any flake re-run alone.
5. **The tests (a)–(g)** each exist and are named in the log. (d)'s bound quotes its provenance.
   (g)'s configuration is at $x_{\rm resp} = 980$, its reference is A2's "seam closed" cell, and its
   tolerance states the reference's own error.
6. **The breakage record.** `HANDOVER_OVERLAP_NODES = 0` fails (c) and the continuity bound; the
   switch at $z_{\rm init}$ fails (b) and (f). Each names its failing tests and was reverted. If a
   record is thin, re-apply the breakage by hand in the worktree, run the module, and revert.
7. **The fixture.** `drop_first_WKB_sample`'s default is unchanged; the numeric samples reach at
   least `HANDOVER_OVERLAP_NODES + 3` nodes below $z_{\rm init}$; every existing test whose numbers
   moved is listed in the log with why.
8. **The records.** The log has every heading of the previous handover logs and classifies each
   deviation (the Gate's exception, and the worktree, among them). `HANDOVER-B1.md` exists. The
   board's B1 row and item; the `realistic_large_x.py` observation opened as an issue with
   `docs/OPEN_ISSUES.md` to match, count and date correct by counting. `black --check` is clean on
   every touched `.py` file. The commit message follows `CLAUDE.md`'s form.

### A.4 Landing

Only when **both** hold:
- 02's commit is on `handover-remedial`, and its orchestrator's review has finished;
- `runregistrykit list` shows no `levin-phase-floor-02-*` run whose pid answers `kill -0`.

Then, in the worktree: `git rebase handover-remedial`, resolving a conflict on the board or the
index by reading both sides. If the rebase brought any `.py` file outside `docs/`, re-run the seven
roots. Then, in the main checkout, `git merge --ff-only handover-12-b1`, and push. The landed SHA
goes on the board in the orchestrator's residue commit. Remove the worktree and the branch once
landed.

If 02 has stopped and is waiting on the user, do not land: ask the user.

---

## B. Part B — rehearsal R8, on the Mac Studio

### B.0 When it starts

Only when **all** hold, each checked by the laptop session that dispatches it:
- Part A has landed and is pushed;
- `levin-phase-floor` 03's Mac Studio orchestrator has ended. Its records need not be committed
  yet: `var/levin-phase-floor-03/` is untracked, and moving the checkout does not touch it;
- on the Mac Studio: `runregistrykit list` shows nothing `running` or stale; the lock is free;
  `git status --short` is empty.

The laptop then fast-forwards the Mac Studio's checkout to the landed SHA (`git merge --ff-only
<SHA>` after a fetch, over `ssh macstudio-tunnel`) and starts the Mac Studio orchestrator, a
Claude Code session over the saved `MacStudio` SSH connection, with a dated addendum to this
section recording what it checked. That orchestrator reads this §B and the prompt's §6.

### B.1 Facts, as at the dispatch SHA

Re-checked at R8's dispatch and corrected in the addendum where they have moved.
- **R3's recorded runs** are on the Mac Studio at `var/handover-06/recorded/`:
  `rrk-v020-adoption-03-reference-20261018T232838` and
  `rrk-v020-adoption-03-resumed-c-20261018T233941`. `compare_with_recorded.py`'s `--against` names
  the first.
- **The launch** is R7's (`docs/py313-venv-verification.md` §2.1), with this prompt's names:
  ```bash
  nohup sh -c "PYTHONPATH=. ./venv/bin/python -u RunDrivers/quadsource_atol_sweep.py \
      --build --rehearsal \
      --database var/datastores/handover-12-reference.sqlite \
      --job-name handover-12 --atol 1e-32 --rtol 1e-7 --cpus 6 \
      --register r8-reference --purpose \"\$0\" --campaign handover --prompt 12; \
    echo \"** driver exit status \$?\"" "<one line>" \
    > var/runs/handover-12-reference.launch.log 2>&1 &
  disown
  ```
  The `--purpose` names R8 of `handover` prompt 12 on the Mac Studio, the landed SHA, and that the
  fingerprint is expected to **differ** from R3's `39809dca…` in `QuadSource` and
  `QuadSourceIntegral` only. It names no temp dir.
- **The lock:** `acquire --project SecondaryGWKit --purpose "<R8, the SHA>" --ray-temp-dir
  /tmp/ray-sgk --ray-address in-process`, then `set-pid` with the driver's Python pid, held
  through the readers, then `release` once `ray-processes` is empty. Each a shell command of its
  own, output by `>>`, never a pipe.
- **Token authentication** is on (`config/ray_local.py`); `~/.ray/auth_token` exists at mode 600.
  Never print it.
- **`--cpus 6`**, never more: the Mac Studio serves a website's object store.
- **The expected difference.** `compare_with_recorded.py` exits 1, because the fingerprints
  differ. The check is **which classes** differ: `QuadSource` and `QuadSourceIntegral`, and nothing
  else. A difference in any other class is prompt §8's stop.
- **The readers:** `bash docs/dk-and-rpk-migration/rehearsal/run_readers.sh
  var/datastores/handover-12-reference.sqlite var/handover-12/extract`, with the store's files
  hashed before and after.
- **The roots** on the Mac Studio at the landed SHA: `ComputeTargets` and `ObjectFactories` equal
  the laptop's counts at Part A's review.

### B.2 Dispatch

One fresh-context subagent on **Opus**, in the Mac Studio's checkout, given the prompt (its §6 is
its scope), the landed SHA and §B.1 with the addendum. It commits nothing and runs no git command
that writes. It writes only under `var/handover-12/`, the R8 store and its sidecar, its run
directory and launch log, the lock and `/tmp/ray-sgk`. Launch, one check, end the turn; the
orchestrator waits on the pid once in the background and resumes it with the final `status.json`.
Its §6.5 analysis scripts go under `var/handover-12/scripts/`, and its records are
`var/handover-12/{out,scripts,state}`, `log.md` (§6's part, to be appended to the code log),
`verification.md` (for `docs/handover/HANDOVER-B1.md`'s R8 section) and `issues.md`.

### B.3 The review — on the Mac Studio

1. **Scope.** `git status --short` empty, `HEAD` the landed SHA; `find var -newer <marker>` lists
   only §B.2's paths. A3 v2's five files and its sidecar are unchanged.
2. **The run.** `done`, exit 0; the manifest names `handover`, `12`, `r8-reference`, `git_dirty:
   false`, the landed SHA, Python 3.13.16.
3. **The fingerprint.** `store fingerprint` without `--write`, by you; `compare_with_recorded.py`
   against R3's `reference`, run by you: the differing classes are `QuadSource` and
   `QuadSourceIntegral` and no other. Record the new digest as R8's.
4. **The rows.** By a scratch reader, `mode=ro`: every `QuadSourceIntegral` row has
   `partition.handover` and no `clamp_gaps_log1pz` or `max_clamp_gap_log1pz`; the `QuadSource`
   rows' extent against R3's, with the nodes added.
5. **The readers** all exit 0, and the store's files are identical before and after.
6. **The lock.** The transcript: `acquire` 0, `set-pid` 0 with the driver's pid, `release` 0.
   `ray-processes` lists nothing and `status` says free, each run by you.
7. **The records** as §B.2 lists them; the log classifies each deviation.

The continuity residuals are recorded, not judged (prompt §6).

### B.4 After the review — on the laptop

The laptop copies `var/handover-12/` back (no CSV, no store) and makes B1's **records commit**:
§6's record appended to `logs/12-remove-the-tk-gap.md`, R8's section of `HANDOVER-B1.md`, the
board's B1 row and item with R8's fingerprint, the header's three issues moved to the
`source-remediation` board's §4 and `[06-source-spline-residual-vs-handover]` narrowed, and
`docs/OPEN_ISSUES.md` to match, count and date corrected by counting; this directory's README row
for 12. R8's store is the user's to keep or retire; ask.

---

## 5. Stop and ask the user

Relay verbatim; do not adjudicate. The prompt's §8 conditions, or any failed check in §A.3 or §B.3:
report it, do not repair it. **Never write to A3 v2**, never move the Mac Studio's checkout while
03 or R8 runs, and never end another session's run.
