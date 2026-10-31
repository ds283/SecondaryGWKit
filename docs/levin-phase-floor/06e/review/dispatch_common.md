## Common to both Part B agents of prompt 06e (levin-phase-floor campaign), on the Mac Studio

You are one of two fresh-context agents running Part B of prompt 06e of the `levin-phase-floor`
campaign of SecondaryGWKit, on the Mac Studio (`mps026298.phys.susx.ac.uk`). An orchestrator
session dispatched you; it reviews your records afterwards. Today is 2026-10-30.

**Your prompt** is `prompts/levin-phase-floor/06e-exact-end-nodes.md`, read in the code worktree
(below). Read it in full (§0–§3 explain the code you are measuring), and the `CLAUDE.md` of the
code worktree, which binds you (the run registry, the shared rehearsal lock, the venv, `black`).
Your scope is the section named in your own brief below; the other section is the other agent's.
Prompt 06e says "as 06c §4" and "06c's census": read
`prompts/levin-phase-floor/06c-global-error-control.md` §4 and §5 for that machinery, and 06c's
Part B records in `docs/levin-phase-floor/06c/rehearsal/` and `…/06c/census/` (and
`…/06c/depth20-tiling/`) for how it was done. Read the code log
`prompts/levin-phase-floor/logs/06e-exact-end-nodes.md` ("Observations not acted on" and "State
handed on") as context. **Do not read anything under `prompts/levin-phase-floor/orchestrator/`**,
nor `docs/levin-phase-floor/06c/review/`.

### Where things are

- **Part B's dispatch SHA** is `6d81fad43cbf1f17cff8d0f48fc7b65966e343ab` (`6d81fad`). It adds only
  records to 06e's code commit `f4d8e17377e7c82ca3a8096a03343d2abdf38a5c` (`f4d8e17`).
- **The code commit's parent** is `50277f3e019bb37df6d91a5c2340877755498508` (`50277f3`), whose
  code is 06d's (06c's plus 06d's), the code 06c's census ran.
- **06c's Part B dispatch commit** (06c's census tree) is
  `e5764a6708757cb11704b95bcd66825baab4ba29` (`e5764a6`): `git diff --stat e5764a6 50277f3 -- .
  ':!docs' ':!prompts'` shows only `CLAUDE.md` (re-checked by the orchestrator today).
- **The main checkout,** `/Users/ds283/Documents/Code/SecondaryGWKit`, at `handover-remedial`
  `28ebaf0`, holds the venv, the stores, `var/` and the run registry. The records go in its
  `var/levin-phase-floor-06e/`. **No tree runs from the main checkout, and it is never moved.**
- **The parent worktree,** `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06e-50277f3`, detached at
  `50277f3`: R19 and Z0's reproduction of 06c's census units run here.
- **The code worktree,** `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06e-6d81fad`, detached at
  `6d81fad`: R20, the tree $g$ of §5, the value comparison, the readers and the Z3 references run
  here. The prompts are read here.
- The worktrees `SecondaryGWKit-lpf06c-e5764a6` and `-fef401c` are 06c's: **never write to or run
  from them.** Read 06c's tree from git history with `git --no-optional-locks`, if you need it.
- **In both 06e worktrees:**
  - every command calls `/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python` by that
    absolute path, with `PYTHONPATH` set to that worktree and `-B` (or
    `PYTHONDONTWRITEBYTECODE=1`);
  - nothing is written in either worktree, no `__pycache__` included.
- **Stores' paths are absolute,** into the main checkout's `var/datastores/`. Run directories are in
  the main checkout's `var/runs/` (a run launched from a worktree is recorded there).
- **No git command that writes,** in any tree, `fetch` included. Read git with
  `git --no-optional-locks`. You commit nothing; the laptop copies `var/levin-phase-floor-06e/`
  back and commits it.
- **`HEAD` is not yours.** Do not assume any tree's `HEAD` reflects your work; check each tree's
  `HEAD` and `git status --short` where you rely on them.

### Facts (from earlier records; check what you rely on — no fact here is to be trusted over the code or the store)

- **Re-checked by the orchestrator on the Mac Studio at 07:40 UTC today, read-only** (after the
  laptop's own check at 07:37):
  - the main checkout is `handover-remedial` at `28ebaf0`, `git status --short` empty; both 06e
    worktrees are clean (`git status --short --ignored` empty) at `50277f3` and `6d81fad`;
  - the registry holds 31 runs, all finished, nothing `running` or stale;
  - the lock is free (`~/.raypoolkit/rehearsal-lock` does not exist), and `ray-processes
    --ray-temp-dir /tmp/ray-sgk` lists none. `status` takes `--ray-temp-dir` and `--ray-address`
    but **not** `--project`;
  - there is no `var/levin-phase-floor-06e*` (other than the orchestrator's own
    `var/levin-phase-floor-06e/review/`) and no `var/datastores/levin-phase-floor-06e*`;
  - the scratch copy's five files match `copy_files_certified.txt` by SHA-256, with no journal,
    `-wal` or `-shm` file beside them;
  - 06c's census `checkpoint.jsonl` has SHA-256
    `b61cb034c1ee95217fae6edb3d14ace0193d2ff88ce66ee0bd642099873012f7`;
  - `~/.ray/auth_token` is mode 600; 96 GiB free. **The load is about 12**, from macOS and desktop
    processes (`mds_stores` at ~90% of one core, ChronoSync, `fileproviderd`, WindowServer), with no
    Python of this project running. Leave them alone; if the load bears on the 300 s budget, record
    it as an observation.
- **A3 v2's scratch copy** is `var/levin-phase-floor-06/wider/state/a3v2_copy/` (five files). Its
  certification is `var/levin-phase-floor-06/wider/state/copy_files_certified.txt`: primary
  `1542e05c…`; shards `bb007a4c…`, `aae63ccd…`, `5e2bda9b…` and `ecc3330d…`. Its fingerprint and
  A3 v2's are both `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`.
- **06c's Part B** ran on this machine on 2026-10-29. Its records are committed in
  `docs/levin-phase-floor/06c/` (read them in the code worktree); the uncommitted rest is in the
  main checkout's `var/levin-phase-floor-06c/` (`docs/levin-phase-floor/06c/KEPT_ON_MAC_STUDIO.md`
  lists it). **Read it only.**
  - **R17 and R18** (`levin-phase-floor-06c-r17-parent-20261029T123720`,
    `…-r18-global-20261029T124058`) both gave `1b332ffa…`. Their launches are in
    `docs/levin-phase-floor/06c/rehearsal/state/launch_r17.txt` and `launch_r18.txt`.
  - **06c's census run** `var/runs/levin-phase-floor-06c-census-20261029T130214/` (1.3 GB) is the
    "before" of §5, read from its `checkpoint.jsonl` (`b61cb034…`) and its unit pickles. Its unit
    wall was 3.21 h at `1e-7` and 5.77 h at `1e-8`.
  - 06c's census harness is `docs/levin-phase-floor/06c/census/scripts/`, byte-identical to the run.
  - `depth20-tiling`'s 30 sampled calls, their captured inputs and Y1-method reference machinery are
    in `docs/levin-phase-floor/06c/depth20-tiling/` (committed) and
    `var/levin-phase-floor-06c/depth20-tiling/` (state, captures, `ref/`), with the orchestrator's
    `orch_ref.py` in `docs/levin-phase-floor/06c/depth20-tiling/orchestrator_check/`.
- **R8's fingerprint,** which R19 and R20 must both give, is
  `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`. R9–R18 gave it.
- 05's `classify_non_qsi.py` is in `docs/levin-phase-floor/05/rehearsal/scripts/`. The readers are
  `docs/dk-and-rpk-migration/rehearsal/run_readers.sh`; 06c's worktree adaptation is
  `docs/levin-phase-floor/06c/rehearsal/scripts/run_readers_worktree.sh`.
- **"Value-identical" means the computed values.** It excludes timings, labels, `parents_digest`
  and work counters (`[05-store-value-diff-compares-volatile-columns]`).
- **Every `QuadSourceIntegral` value is expected to move.** The store fingerprint certifies
  identities, tags and row counts, not values (`CLAUDE.md` rule 7).

### The addendum the orchestrator's note records at Part B's dispatch (verbatim)

### B.2 Addendum, 2026-10-30: Part B's dispatch (§B.1)

**Part A** is the code commit **`f4d8e17`**, whose parent is **`50277f3`**, reviewed (§A.5) and
pushed.

**The trees, by SHA:**
- **R19 and Z0's reproduction** at **`50277f3`**, in the parent worktree. Its code is 06d's, which
  06c's census ran: `git diff --stat e5764a6 50277f3 -- . ':!docs' ':!prompts'` shows only
  `CLAUDE.md` (checked on the laptop), so Z0's first clause holds against 06c's Part B dispatch
  tree `e5764a6`.
- **R20 and $g$** at **this addendum's commit**, in the code worktree. Its code is `f4d8e17`'s:
  the commit adds only records.
- **06c's census** `levin-phase-floor-06c-census-20261029T130214` is the "before", its
  `checkpoint.jsonl` `b61cb034…` (`docs/levin-phase-floor/06c/KEPT_ON_MAC_STUDIO.md`).

**What Part B must carry:**
- **The cost should fall.** On the laptop's draws depth-20 calls fell 101 → 0 and solves
  3.38 M → 0.47 M; the depth-20 sample's wall time fell ×26 with the ends pinned. A census longer
  than 06c's (3.21 h and 5.77 h of unit wall at `1e-7` and `1e-8`) is itself a finding.
- **A fallback solve costs about twice as much** with θ′ supplied (97 → 186 µs, N − 1 extra θ′
  evaluations); both production callers supply θ′.
- **η_θ = 2⁻¹⁰ was measured on closed-form θ′ only.** Production's θ′ is the first test of its
  margin; a Z3 or Z4 failure on a Clenshaw–Curtis-dominated call should be read against it.
- **Z3:** 06d was dishonest wherever the end-node offset happened to hide (06e's log, Observations
  2), so the census side, $|v_{06c} - {\rm ref}|/{\rm abserr}_{06c}$, may exceed 1 on references
  where $g$ is honest.
- **The near-singular class** may surface converged and dishonest if A3 v2 reaches it
  (`[06c-budget-can-stop-on-a-pre-asymptotic-residual]`, widened by 06e).

### Six cores, across everything

**At most six cores are in use at any time on this host, across both agents and the
orchestrator.** So the work is strictly sequential, and each stage starts only when the
orchestrator tells you:
1. R19 (Ray, `--cpus 6`), then R20 (Ray, `--cpus 6`);
2. the rehearsal agent's comparison, Z1 and readers, then the lock's release;
3. the census agent's Z0, then the census (six workers);
4. the Z3 references (six workers each), one at a time.

While another stage runs, an agent does no computation beyond reading files, copying the harness,
diffing and writing notes. Hashing the scratch copy or a store, Z0, fingerprinting, the value
comparison and every unit or reference run count as computation.

### Rules for both of you

- **Never write to** A3 v2 (`var/datastores/handover-A3-v2-lambdacdm*`), its scratch copy (read it
  only, opened read-only), or any earlier prompt's records, stores or runs (`var/levin-phase-floor-0*`
  other than `-06e`, `var/datastores/levin-phase-floor-0*` other than `-06e-r19*`/`-06e-r20*`,
  `var/runs/` other than your own runs). That includes 06c's census run directory and
  `var/levin-phase-floor-06c/depth20-tiling/`, which you only read. Never write in
  `var/levin-phase-floor-06e/review/` (the orchestrator's). Never end another session's run. Never
  touch anything of prompt 07's (`var/levin-phase-floor-07*`, any stash).
- **Ray.** No Ray session may be started by anything but R19's and R20's own drivers. Production
  modules import the `ray` package, and that is fine; what must never happen outside R19/R20 is Ray
  being **initialised** (a Ray session appearing). Every Ray driver names its address; `ray stop`
  is never run.
- **Save every command you run and its output** under your `out/` (or `state/` for launch
  transcripts). The review reads the outputs, not your log.
- **Launch, one check, end the turn** (`CLAUDE.md`, run registry rule 4). Launch detached
  (`nohup sh -c …`, manifest first, stdout/stderr into the run directory or your records, pid
  recorded), verify in **one** check that it started, then end your turn and report: the run id
  (if registered), the run directory, the detached Python pid, the `sh` pid and the first
  checkpoint/heartbeat line. **Do not poll, do not `sleep`-wait, never `pgrep -f`.** The
  orchestrator waits on the pid and resumes you with the final `status.json` (or the exit).
- **Every deviation from the prompt is numbered and classified** (`STRUCTURALLY REQUIRED`,
  `IMPLEMENTATION CHOICE` or `UNINTENDED DRIFT`) before you report, failed attempts included.
- **Stop and ask** at any of prompt 06e §7's conditions that apply to your section, and at anything
  this brief or the prompt does not settle that changes what you measure. To stop, end your turn
  with the question stated plainly **as its first line**, and the state of every file and run you
  have made. Do not guess and do not repair.
- **Correctness is the only objective.** An argument from cost or time for a shortcut is a
  deviation to record, not a reason to take it.
- **Your records** go in your directory under the main checkout's `var/levin-phase-floor-06e/`:
  - `log.md` — your section, in campaign README §5.1's form
    (`prompts/levin-phase-floor/README.md` §5.1), **every heading of it** (including **What
    shipped**), for appending to `logs/06e-exact-end-nodes.md`;
  - `verification.md` — your part of `docs/levin-phase-floor/VERIFICATION.md` §6e, "Exact end
    nodes", with links in the `06e/…` form (the laptop commits your directory as
    `docs/levin-phase-floor/06e/rehearsal/` or `…/06e/census/`);
  - `issues.md` — anything to open on the board, or "none";
  - `out/`, `scripts/` and `state/`.
