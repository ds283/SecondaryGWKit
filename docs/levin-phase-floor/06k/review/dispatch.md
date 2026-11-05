## Prompt 06k of the levin-phase-floor campaign, on the Mac Studio — your brief

You are the one fresh-context agent running prompt 06k of the `levin-phase-floor` campaign of
SecondaryGWKit, on the Mac Studio (`mps026298.phys.susx.ac.uk`). An orchestrator session dispatched
you, and it reviews your records afterwards. Today is 2026-11-04 (UTC).

**Your prompt** is `prompts/levin-phase-floor/06k-slow-leaf-trust-routes.md`, read in the worktree
(below). Read it in full, and everything its "Read first" names. Also read the worktree's
`CLAUDE.md`, which binds you (the run registry, the venv, `black`), and the campaign README
`prompts/levin-phase-floor/README.md` §4.1 and §5 (§5.1 is the log's form).

**Do not read** anything under `prompts/levin-phase-floor/orchestrator/`,
`docs/levin-phase-floor/*/review/`, any `docs/levin-phase-floor/*/review.md`, or the main checkout's
`var/levin-phase-floor-06k/review/` beyond this brief. **One exception:** your prompt names
`docs/levin-phase-floor/06g/review.md` "for the census's cost"; read that file.

### Where things are

- **The dispatch SHA** is `89370609a64d58993289dcfe991350a420ea698d` (`89370609`), on
  `origin/handover-remedial`, with 06j's records commit `6e8c2559` before it. `git diff --stat
  1c5336e 89370609 -- . ':!docs' ':!prompts'` shows only `CLAUDE.md`, `tools/filon_reference.py`
  and `tools/tests/test_filon_reference.py` (re-checked by the orchestrator at 08:47 UTC today).
- **The worktree,** `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06k-8937060`, detached at
  `89370609`, with no venv. `git status --short --ignored` was empty at 08:47 UTC. All your code
  runs here; the prompts are read here. **Nothing is written in it,** `__pycache__` included.
- **The main checkout,** `/Users/ds283/Documents/Code/SecondaryGWKit`, `handover-remedial` at
  `28ebaf07`, clean. It holds the venv, `var/` and the run registry. **No tree runs from it, and it
  is never moved.**
  - The scratch copy: `var/levin-phase-floor-06/wider/state/a3v2_copy/` (five files; certification
    `var/levin-phase-floor-06/wider/state/copy_files_certified.txt`; fingerprint `0e32f24a…`).
  - 06g's records: `var/levin-phase-floor-06g/` (read only).
  - 06g's census run: `var/runs/levin-phase-floor-06g-census-20261101T114315/` (read only).
  - 06h's census run: `var/runs/levin-phase-floor-06h-census-20261101T005037/` (read only).
  - 06j's records: `var/levin-phase-floor-06j/` (read only; its capture pickles were deleted).
  - **Your records** go in `var/levin-phase-floor-06k/` (`out/`, `scripts/` with
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
  `git --no-optional-locks`. You commit nothing; the laptop copies `var/levin-phase-floor-06k/` back
  and commits it. **`HEAD` is not yours:** check each tree's `HEAD` and status wherever you rely on
  them.

### The orchestrator's note, §B.0, verbatim

> ### B.0 Facts, checked by the orchestrator on the laptop, 2026-11-04
>
> - **06g's census** `levin-phase-floor-06g-census-20261101T114315` is kept in full on the Mac Studio.
>   - Its checkpoint is `03adcbba8a78d82d9d6b899b8eee8b9ae1659763f52deb066d2840af3d82c0fa`, with 16
>     region pickles.
>   - It took 5 h 40 min. 06h's took 2 h 33 min.
>   - `census/out/v8_calls.json` (`1e9a62b0…`) lists V8's lost calls.
> - **The references** are committed:
>   - 06g's: `docs/levin-phase-floor/06g/references/out/scores_stage3/scores.json`, `rows` (102:
>     V3's 90 and V8's 12) and `V1` (12);
>   - 06j's: `docs/levin-phase-floor/06j/out/w_calls.csv`, 22 calls on 17 references.
>   06g's reference pickles were deleted on 2026-11-03. No route needs a new reading.
> - **06j's 46 captures were deleted on 2026-11-03.** They are reproducible: run 1, which made them,
>   took 11 minutes, and J0 showed each equal to 06g's census bit for bit.
> - **06j's replays** took 0.2–10 s per call (103 s for all 46).
> - **The scratch copy** `var/levin-phase-floor-06/wider/state/a3v2_copy/` is certified; its
>   fingerprint is `0e32f24a…`.
> - **The Mac Studio** had 89 GB free on 2026-11-03. Three census run directories at about 4.5 GB each
>   need about 14 GB.
> - **06g's four worktrees** `SecondaryGWKit-lpf06g-{1c5336e,4f4f9ed,b5626ec,ff3e839}` stay, on the
>   user's word; neither you nor the agent touches them.
> - **No Ray, no lock, no store built.** If anything needs Ray, that is a stop.

### Re-checked by the orchestrator on the Mac Studio, 08:47–08:49 UTC today (read-only)

- The registry holds 48 runs, none `running` or stale (42 done, 2 failed, 4 killed). `~/.raypoolkit/`
  is empty; the lock is free; `ray-processes --ray-temp-dir /tmp/ray-sgk` lists none.
- The scratch copy's five files match `copy_files_certified.txt` by SHA-256, with no journal, `-wal`
  or `-shm` beside them.
- 06g's census checkpoint is `03adcbba…` with 16 region pickles; `v8_calls.json` is `1e9a62b0…`,
  both by SHA-256.
- There was no `var/levin-phase-floor-06k/` before the orchestrator's `review/`.
- The host has 10 cores and 89 GB free on disk. No Python of this project was running. The load
  average was high (about 27) from system services (WindowServer, QuickLook, ChronoSync,
  Spotlight); bear it in mind when you time things.

You make the prompt's Gate checks yourself (the fingerprint included); these facts do not replace
them.

### Rules

- **Never write to:** A3 v2 or any store; the scratch copy (read only, opened read-only); any
  earlier prompt's records or runs (`var/levin-phase-floor-0*` other than `-06k`, and `var/runs/`
  other than your own runs); `var/levin-phase-floor-06k/review/`. Never end another session's run.
  **Never touch anything of prompt 07's** (`var/levin-phase-floor-07*`, any stash).
- **No change to the tree.** The variants live under `var/levin-phase-floor-06k/scripts/variants/`
  and are loaded in place of the tree's module by your harness, in its own processes.
- **No Ray, no lock, no store built.** Production modules import `ray`, and `ray.cloudpickle` as an
  unpickler is fine; what must never happen is a Ray session being initialised. If anything needs
  Ray, that is a stop. `ray stop` is never run.
- **At most six processes** of computation at any time, including any analysis you run while a
  census runs.
- **Save every command you run and its output** under `out/`, or under `state/` for launch
  transcripts. The review reads the outputs, not your log.
- **Write the K0 draw rule and its seed, and copy the predictions L1, L2, C1, C2, H1, S1 and K1
  verbatim into the log, before the first number** (before K0's first unit is run). The review
  checks this by the files' timestamps, so write them as files and do not rewrite them afterwards
  (a correction is a new file, and a deviation).
- **Planned stop 1 — after Gate K0 and Part L, before the first census.** Stop and return to the
  orchestrator with:
  - each variant's path, its saved diff against the tree's `AdaptiveLevin/levin_quadrature.py`,
    and your AST check (only `_adaptive_levin` differs, and within it only what prompt §0.2 names);
  - K0's records: the identity variant on all 100 units (counters included), and each route's
    zero-firing units against 06g;
  - Part L's tables and the LiouvilleGreen draws' tables, with where they are;
  - **the census plan:** the launch command and run order; the cost projection for each census,
    from 06g's (5 h 40 min) and 06h's (2 h 33 min) walls; and the backstop, which per the prompt
    reports a run past its projection with its progress and projected finish, and **stops it only
    at a projected day or more, counting the units in flight, not only those finished**.

  The orchestrator checks it and resumes you. Launch no census before it does.
- **Planned stop 2 — between censuses.** Launch one census at a time. When it finishes, the
  orchestrator checks that it is `done`, 18,560 of 18,560, and resumes you; only then launch the
  next.
- **Launch, one check, end the turn** (`CLAUDE.md`, run registry rule 4). Anything that may exceed
  about ten minutes is registered (`--campaign levin-phase-floor --prompt 06k`) and launched
  detached (`nohup sh -c …`, manifest first, stdout and stderr into the run directory, pid recorded).
  Verify in **one** check that it started, then end your turn and report the run id, the run
  directory, the detached Python pid, the `sh` pid and the first checkpoint or heartbeat line.
  **Do not poll, do not `sleep`-wait, never `pgrep -f`.** The orchestrator waits on the pid and
  resumes you with the final `status.json` (or the exit).
- **Stop and ask** at any of prompt 06k §7's conditions, and at anything this brief or the prompt
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
  over files anywhere outside `var/levin-phase-floor-06k/`.

### Your records

In the main checkout's `var/levin-phase-floor-06k/`:
- `log.md`, in campaign README §5.1's form with **every heading of it**, for the laptop to copy to
  `prompts/levin-phase-floor/logs/06k-slow-leaf-trust-routes.md`;
- `verification.md`, the text of `docs/levin-phase-floor/VERIFICATION.md` §6k, with links in the
  `06k/…` form (the laptop commits your directory as `docs/levin-phase-floor/06k/`);
- `issues.md`: anything to open on the board, or "none";
- `out/` (with the variants' diffs and AST checks, K0's checks, Part L's and the draws' tables,
  each census's comparison tables and reference scores, and `out/results.md`, which states each
  route's verdicts with their numbers and what the route would do as a remedy, compares the three
  side by side, and **proposes no code**), `scripts/` (with the diff against 06g's and 06j's
  copies, and `variants/`) and `state/`;
- a provenance header on every output (README §5 rule 5);
- `KEPT_ON_MAC_STUDIO.txt`: every pickle, every census run directory and every file over 2 MB you
  made, with SHA-256 and size.

When you finish, end with a short report: K0, and L1, L2, C1, C2, H1, S1 and K1 per route with
their numbers, the cost of each census, every deviation, every run id, and anything left running
(there should be nothing).
