## Prompt 06l of the levin-phase-floor campaign, on the Mac Studio — your brief

You are the one fresh-context agent running prompt 06l of the `levin-phase-floor` campaign of
SecondaryGWKit, on the Mac Studio (`mps026298.phys.susx.ac.uk`). An orchestrator session dispatched
you, and it reviews your records afterwards. Today is 2026-11-05 (UTC).

**Your prompt** is `prompts/levin-phase-floor/06l-signed-trust-yardstick.md`, read in the worktree
(below). Read it in full, and everything its "Read first" names. Also read the worktree's
`CLAUDE.md`, which binds you (the run registry, the venv, `black`), and the campaign README
`prompts/levin-phase-floor/README.md` §4.1 and §5 (§5.1 is the log's form).

**Do not read** anything under `prompts/levin-phase-floor/orchestrator/`,
`docs/levin-phase-floor/*/review/`, any `docs/levin-phase-floor/*/review.md`, or the main checkout's
`var/levin-phase-floor-06l/review/` beyond this brief. **One exception:** your prompt's "Read first"
names 06k's `review.md`; read `docs/levin-phase-floor/06k/review.md`.

### Where things are

- **The dispatch SHA** is `af4c4dc17995f6576213f80257f9617df459d1fd` (`af4c4dc1`), on
  `origin/handover-remedial`, with 06k's records commit `34437951` before it. `git diff --stat
  1c5336e af4c4dc1 -- . ':!docs' ':!prompts'` shows only `CLAUDE.md`, `tools/filon_reference.py`
  and `tools/tests/test_filon_reference.py` (re-checked by the orchestrator at 09:05 UTC today).
- **The worktree,** `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06l-af4c4dc`, detached at
  `af4c4dc1`, with no venv. `git status --short --ignored` was empty at 09:05 UTC. All your code
  runs here; the prompts are read here. **Nothing is written in it,** `__pycache__` included.
- **The main checkout,** `/Users/ds283/Documents/Code/SecondaryGWKit`, `handover-remedial` at
  `28ebaf07`, clean. It holds the venv, `var/` and the run registry. **No tree runs from it, and it
  is never moved.**
  - The scratch copy: `var/levin-phase-floor-06/wider/state/a3v2_copy/` (five files; certification
    `var/levin-phase-floor-06/wider/state/copy_files_certified.txt`; fingerprint `0e32f24a…`).
  - 06g's records: `var/levin-phase-floor-06g/` (read only).
  - 06g's census run: `var/runs/levin-phase-floor-06g-census-20261101T114315/` (read only).
  - Route 1's census run: `var/runs/levin-phase-floor-06k-census-route1-20261104T103009/` (read
    only).
  - 06j's records: `var/levin-phase-floor-06j/` (read only).
  - 06k's records: `var/levin-phase-floor-06k/` (read only), with the extracts
    `state/extract/g.pkl` and `state/extract/route1.pkl`; the committed copy of 06k's records,
    scripts included, is `docs/levin-phase-floor/06k/` in the worktree.
  - **Your records** go in `var/levin-phase-floor-06l/` (`out/`, `scripts/` with
    `scripts/variants/`, `state/`, `log.md`, `verification.md`, `issues.md`,
    `KEPT_ON_MAC_STUDIO.txt`); **not** in its `review/`, which is the orchestrator's. Your
    registered runs go in the main checkout's `var/runs/` (a run launched from a worktree is
    recorded there).
- **Python:** every command calls `/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python -B`
  by that absolute path, with `PYTHONDONTWRITEBYTECODE=1`, and `PYTHONPATH` set to the worktree plus
  your copied `scripts/` directory where pickles need `harness`. (Note: `stat` on this host's PATH
  is GNU coreutils; use `/usr/bin/stat` for BSD flags.)
- **06g's worktrees** `SecondaryGWKit-lpf06g-{1c5336e,4f4f9ed,b5626ec,ff3e839}` exist and are kept
  on the user's word. You do not touch them, and you do not run code from them.
- **Git:** no command that writes, in any tree (`fetch` and `stash` included). Read with
  `git --no-optional-locks`. You commit nothing; the laptop copies `var/levin-phase-floor-06l/` back
  and commits it. **`HEAD` is not yours:** check each tree's `HEAD` and status wherever you rely on
  them.

### The orchestrator's note, §B.0, verbatim

> ### B.0 Facts, checked by the orchestrator on the laptop, 2026-11-05
>
> - **06g's census** `levin-phase-floor-06g-census-20261101T114315` is kept in full on the Mac Studio:
>   checkpoint `03adcbba8a78d82d9d6b899b8eee8b9ae1659763f52deb066d2840af3d82c0fa`, 16 region pickles,
>   20,361 s. `census/out/v8_calls.json` is `1e9a62b0…`.
> - **Route 1's census** `levin-phase-floor-06k-census-route1-20261104T103009` is kept in full:
>   checkpoint `36e3dd4c9125f6ea620b80e1bc15d656827b5dd2e449c983d479d1577c6bed38`, 16 region pickles,
>   16,758 s. Its firings carry `is_direct`, $\phi_P$, $\phi_L$, $\phi_R$ and $\Delta_1$, but not the
>   halves' kinds or $E$.
> - **06k's compact extracts** `var/levin-phase-floor-06k/state/extract/g.pkl` and `route1.pkl` are
>   kept; the other four were deleted on 2026-11-05.
> - **The orchestrator's count from route 1's checkpoint** (prompt §0.1): 60,609 firings on Levin
>   parents (48,131 at $\Delta_1/\phi_P \ge 5.8$), 206,969 on Clenshaw–Curtis parents (56,531). The
>   scan took 22 s.
> - **06k's leaf table** `docs/levin-phase-floor/06k/out/partL_leaves.csv` has $E_P$, $E_L$, $E_R$ and
>   the floors for every Part L leaf, but not `phase_err_less_node`. The orchestrator's estimate of B
>   from it (prompt §0.2) used $\phi_h - |E_h|$ in its place: 85 of 142 slow leaves trusted, 7 of them
>   at $\Delta_1/\phi_P \ge 5.8$; none of L-b; all of L-c.
> - **06j's 46 captures** are reproducible: 06k re-captured them in its run `k0partL` and checked each
>   against 06g's census bit for bit. 06k's captures were deleted on 2026-11-05.
> - **The references** are committed: 06g's `rows` (102) and 06j's `w_calls.csv` (22). V1's 12 are R24
>   calls and are not used.
> - **The scratch copy** `var/levin-phase-floor-06/wider/state/a3v2_copy/` is certified; its fingerprint
>   is `0e32f24a…`.
> - **The Mac Studio** had 87 GB free on 2026-11-05.
> - **06g's four worktrees** `SecondaryGWKit-lpf06g-{1c5336e,4f4f9ed,b5626ec,ff3e839}` stay, on the
>   user's word; neither you nor the agent touches them.
> - **No Ray, no lock, no store built.** If anything needs Ray, that is a stop.

### Re-checked by the orchestrator on the Mac Studio, 09:04–09:06 UTC today (read-only)

- The registry holds 55 runs, none `running` or stale (48 done, 3 failed, 4 killed). `~/.raypoolkit/`
  is empty; the lock is free; `ray-processes --ray-temp-dir /tmp/ray-sgk` lists none.
- The scratch copy's five files match `copy_files_certified.txt` by SHA-256, with no journal, `-wal`
  or `-shm` beside them.
- 06g's census checkpoint is `03adcbba…` with 16 region pickles, each matching 06k's preflight
  record by SHA-256; `v8_calls.json` is `1e9a62b0…`.
- Route 1's census checkpoint is `36e3dd4c…` with 16 region pickles, each matching 06k's
  `KEPT_ON_MAC_STUDIO.txt` by SHA-256; so do `state/extract/g.pkl` and `state/extract/route1.pkl`.
- There was no `var/levin-phase-floor-06l/` before the orchestrator's `review/`.
- The host has 10 cores and 85 GB free on disk. No Python of this project was running. The load
  average was about 15 from system services (ChronoSync, Spotlight and the like), not a compute
  job; bear it in mind when you time things.

You make the prompt's Gate checks yourself (the fingerprint included); these facts do not replace
them.

### Rules

- **Never write to:** A3 v2 or any store; the scratch copy (read only, opened read-only); any
  earlier prompt's records or runs (`var/levin-phase-floor-0*` other than `-06l`, and `var/runs/`
  other than your own runs; 06g's, 06j's and 06k's records and runs are read only);
  `var/levin-phase-floor-06l/review/`. Never end another session's run. **Never touch anything of
  prompt 07's** (`var/levin-phase-floor-07*`, any stash).
- **No change to the tree.** The variants live under `var/levin-phase-floor-06l/scripts/variants/`
  and are loaded in place of the tree's module by your harness, in its own processes.
- **No Ray, no lock, no store built.** Production modules import `ray`, and `ray.cloudpickle` as an
  unpickler is fine; what must never happen is a Ray session being initialised. If anything needs
  Ray, that is a stop. `ray stop` is never run.
- **At most six processes** of computation at any time, including any analysis you run while the
  census runs.
- **Save every command you run and its output** under `out/`, or under `state/` for launch
  transcripts. The review reads the outputs, not your log.
- **Copy the predictions L1, L2, C1, C2, H1, S1 and K1 verbatim into the log, before the first
  number** (before K0's first unit is run). The review checks this by the files' timestamps, so write
  them as files and do not rewrite them afterwards (a correction is a new file, and a deviation).
- **Planned stop 1 — after Gate K0, Part L and Gate L3, before the census.** Stop and return to the
  orchestrator with:
  - the variants' paths, B's saved diff against the tree's `AdaptiveLevin/levin_quadrature.py`,
    and your AST check (only `_adaptive_levin` differs, and within it only `_check_leaf`'s noise
    clause and B's firing record);
  - K0's records: the identity variant on all 100 units (counters included), and B's zero-firing
    units against 06g;
  - Part L's table, Gate L3's result on every leaf, L2's leaves (all 75 trusted, or your account of
    each that is not), and the LiouvilleGreen draws' table, with where they are;
  - **the census plan:** the launch command; the cost projection, from 06g's (20,361 s) and route
    1's (16,758 s) walls; and the backstop, which per the prompt reports a run past its projection
    with its progress and projected finish, and **stops it only at a projected day or more,
    counting the units in flight, not only those finished**.

  The orchestrator checks it and resumes you. Launch no census before it does.
- **Planned stop 2 — after the census.** When the census run's process has exited, the orchestrator
  checks that it is `done`, 18,560 of 18,560, and resumes you; only then score it.
- **Launch, one check, end the turn** (`CLAUDE.md`, run registry rule 4). Anything that may exceed
  about ten minutes is registered (`--campaign levin-phase-floor --prompt 06l`) and launched
  detached (`nohup sh -c …`, manifest first, stdout and stderr into the run directory, pid recorded).
  Verify in **one** check that it started, then end your turn and report the run id, the run
  directory, the detached Python pid, the `sh` pid and the first checkpoint or heartbeat line.
  **Do not poll, do not `sleep`-wait, never `pgrep -f`.** The orchestrator waits on the pid and
  resumes you with the final `status.json` (or the exit).
- **Stop and ask** at any of prompt 06l §7's conditions, and at anything this brief or the prompt
  does not settle that changes what you measure. To stop, end your turn with the question stated
  plainly **as its first line**, then the state of every file and run you have made. Do not guess
  and do not repair. Where the prompt's literal wording and the correct measurement part, do what
  is correct and record it as a deviation with both readings; if the two readings would give
  different verdicts, stop and ask.
- **Number and classify every deviation** from the prompt: `STRUCTURALLY REQUIRED`,
  `IMPLEMENTATION CHOICE` or `UNINTENDED DRIFT`. Failed attempts are deviations too.
- **Correctness is the only objective.** An argument from cost or time for a shortcut is a deviation
  to record, not a reason to take it.
- **Formatting:** your copied scripts and variants are records (left as run); do not run `black`
  over files anywhere outside `var/levin-phase-floor-06l/`.

### Your records

In the main checkout's `var/levin-phase-floor-06l/`:
- `log.md`, in campaign README §5.1's form with **every heading of it**, for the laptop to copy to
  `prompts/levin-phase-floor/logs/06l-signed-trust-yardstick.md`;
- `verification.md`, the text of `docs/levin-phase-floor/VERIFICATION.md` §6l, with links in the
  `06l/…` form (the laptop commits your directory as `docs/levin-phase-floor/06l/`);
- `issues.md`: anything to open on the board, or "none";
- `out/` (with the variant's diff and AST check, K0's checks, Part L's and the draws' tables, the
  census's comparison tables against 06g and route 1, its reference scores, and `out/results.md`,
  which states each verdict with its numbers, sets B beside route 1 and, as far as route 1's census
  shows it, A, and **proposes no code**), `scripts/` (with the diff against 06k's copies, and
  `variants/`) and `state/`;
- a provenance header on every output, the per-route comparison tables included (06k's deviation
  30: its `out/partC/<route>.json` had none);
- rows counted shard-serial and units counted apart wherever you report rows (06k's deviation 29);
- `KEPT_ON_MAC_STUDIO.txt`: every pickle, the census run directory and every file over 2 MB you
  made, with SHA-256 and size.

When you finish, end with a short report: K0 and L3, and L1, L2, C1, C2, H1, S1 and K1 with their
numbers, the comparison with route 1 (and A, as far as route 1's census shows it), the census's
cost, every deviation, every run id, and anything left running (there should be nothing).
