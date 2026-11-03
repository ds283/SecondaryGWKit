## Prompt 06j of the levin-phase-floor campaign, on the Mac Studio — your brief

You are the one fresh-context agent running prompt 06j of the `levin-phase-floor` campaign of
SecondaryGWKit, on the Mac Studio (`mps026298.phys.susx.ac.uk`). An orchestrator session dispatched
you, and it reviews your records afterwards. Today is 2026-11-03 (UTC).

**Your prompt** is `prompts/levin-phase-floor/06j-slow-leaves-and-v2-rows.md`, read in the worktree
(below). Read it in full, and everything its "Read first" names. Also read the worktree's
`CLAUDE.md`, which binds you (the run registry, the venv, `black`), and the campaign README
`prompts/levin-phase-floor/README.md` §4.1 and §5 (§5.1 is the log's form).

**Do not read** anything under `prompts/levin-phase-floor/orchestrator/`,
`docs/levin-phase-floor/*/review/`, any `docs/levin-phase-floor/*/review.md`, or the main checkout's
`var/levin-phase-floor-06j/review/` beyond this brief. **One exception:** your prompt names
`docs/levin-phase-floor/06g/review.md` "for how the references were made and what went wrong with
their backstops"; read that file.

### Where things are

- **The dispatch SHA** is `dfa19d98f45b8501f8c9d314059f7c6e0678c6b8` (`dfa19d98`), on
  `origin/handover-remedial`, with 06g's records commit `fdc3a2bd` before it. `git diff --stat
  1c5336e dfa19d98 -- . ':!docs' ':!prompts'` shows only `CLAUDE.md`, `tools/filon_reference.py`
  and `tools/tests/test_filon_reference.py` (re-checked by the orchestrator at 08:50 UTC today).
- **The worktree,** `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06j-dfa19d9`, detached at
  `dfa19d98`, with no venv. `git status --short --ignored` was empty at 08:50 UTC. All your code
  runs here; the prompts are read here. **Nothing is written in it,** `__pycache__` included.
- **The main checkout,** `/Users/ds283/Documents/Code/SecondaryGWKit`, `handover-remedial` at
  `28ebaf07`, clean. It holds the venv, `var/` and the run registry. **No tree runs from it, and it
  is never moved.**
  - The scratch copy: `var/levin-phase-floor-06/wider/state/a3v2_copy/` (five files; certification
    `var/levin-phase-floor-06/wider/state/copy_files_certified.txt`; fingerprint `0e32f24a…`).
  - 06g's records: `var/levin-phase-floor-06g/` (read only).
  - 06g's census run: `var/runs/levin-phase-floor-06g-census-20261101T114315/` (read only).
  - 06h's census run: `var/runs/levin-phase-floor-06h-census-20261101T005037/` (read only).
  - **Your records** go in `var/levin-phase-floor-06j/` (`out/`, `scripts/`, `state/`, `log.md`,
    `verification.md`, `issues.md`, `KEPT_ON_MAC_STUDIO.txt`); **not** in its `review/`, which is the
    orchestrator's. Your registered runs go in the main checkout's `var/runs/` (a run launched from
    a worktree is recorded there).
- **Python:** every command calls `/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python -B`
  by that absolute path, with `PYTHONDONTWRITEBYTECODE=1`, and `PYTHONPATH` set to the worktree plus
  your copied `scripts/` directory where pickles need `harness`.
- **06g's worktrees** `SecondaryGWKit-lpf06g-{1c5336e,4f4f9ed,b5626ec,ff3e839}` exist and are kept
  on the user's word. You do not touch them, and you do not run code from them.
- **Git:** no command that writes, in any tree (`fetch` and `stash` included). Read with
  `git --no-optional-locks`. You commit nothing; the laptop copies `var/levin-phase-floor-06j/` back
  and commits it. **`HEAD` is not yours:** check each tree's `HEAD` and status wherever you rely on
  them.

### The orchestrator's note, §B.0, verbatim

> ### B.0 Facts, checked by the orchestrator on the laptop, 2026-11-03
>
> - **06g's census** `levin-phase-floor-06g-census-20261101T114315` is complete, kept in full on the
>   Mac Studio (`docs/levin-phase-floor/06g/KEPT_ON_MAC_STUDIO.md`). Its checkpoint is
>   `03adcbba8a78d82d9d6b899b8eee8b9ae1659763f52deb066d2840af3d82c0fa`, with 16 region pickles. 06h's
>   checkpoint is `3104c18ed8b119f1798aec75be98d5b39bce641f3b5623b4c1cdc5b18b5ef372`.
> - **Part S's inputs exist on the Mac Studio only:**
>   - `var/levin-phase-floor-06g/census/out/v8_calls.json` (`1e9a62b0…`);
>   - `…/v8_slow_leaves.json` (`b4643033…`);
>   - the census's region pickles, whose `checked_leaves_06g` give every checked leaf's $r$,
>     $\Delta_1$, $\phi_P$ and span.
>
>   The summary is `docs/levin-phase-floor/06g/census/out/v8_slow_leaves.md`.
> - **Part W's rows:** `census/out/v2_rows.csv.gz` lists V2's rows. The call-level set, 62 calls at
>   `1e-7` and 80 at `1e-8`, with each call's ratio, regions and depth, is
>   `census/out/cmd/51_values_beyond.txt`. Its `G+r` calls move by $7.0$–$7.2\times10^{-33}$ on every
>   row, 06h on 2 regions and 06g on 12–16.
> - **The tool** at `b5626ecb` integrates a stationary or near-stationary Route B segment directly.
>   Its readings on monotone calls are bit-identical to `1c5336e`'s, on both hosts. 06g's references
>   measured QAWO's cost at 0.057–0.10 µs per radian (resolution 1, six workers), about 4× that at
>   resolution 2. FCC's cost does not grow with the span.
> - **The captures** unpickle with the harness's `scripts/` on `PYTHONPATH`, and are checked against
>   the census bit for bit by `check_captures.py`.
> - **No Ray, no lock, no store built.** If anything needs Ray, that is a stop.

### Re-checked by the orchestrator on the Mac Studio, 08:46–08:51 UTC today (read-only)

- The registry holds 46 runs, all finished, nothing `running` or stale. `~/.raypoolkit/` is empty;
  `ray-processes --ray-temp-dir /tmp/ray-sgk` lists none.
- The scratch copy's five files match `copy_files_certified.txt` by SHA-256, with no journal, `-wal`
  or `-shm` beside them.
- 06g's census checkpoint is `03adcbba…` with 16 region pickles; 06h's is `3104c18e…`;
  `v8_calls.json` is `1e9a62b0…` and `v8_slow_leaves.json` is `b4643033…`, all by SHA-256.
- There was no `var/levin-phase-floor-06j/` before the orchestrator's `review/`.
- The host has 10 cores and 89 GB free on disk; no Python was running.

You make the prompt's Gate checks yourself (the fingerprint included); these facts do not replace
them.

### Rules

- **Never write to:** A3 v2 or any store; the scratch copy (read only, opened read-only); any
  earlier prompt's records or runs (`var/levin-phase-floor-0*` other than `-06j`, and `var/runs/`
  other than your own runs); `var/levin-phase-floor-06j/review/`. Never end another session's run.
  **Never touch anything of prompt 07's** (`var/levin-phase-floor-07*`, any stash).
- **No Ray, no lock, no store built.** Production modules import `ray`, and `ray.cloudpickle` as an
  unpickler is fine; what must never happen is a Ray session being initialised. If anything needs
  Ray, that is a stop. `ray stop` is never run.
- **At most six processes** of computation at any time.
- **Save every command you run and its output** under `out/`, or under `state/` for launch
  transcripts. The review reads the outputs, not your log.
- **Write each sampling rule, seed and sample, and copy the predictions S1–S4 and W1–W4 verbatim
  into the log, before the first probe or reading.** The review checks this by the files'
  timestamps, so write them as files and do not rewrite them afterwards (a correction is a new file,
  and a deviation).
- **Planned stop before any reference reading.** Before you launch the first Part W reference
  reading (QAWO or FCC, any resolution), stop and return to the orchestrator with your plan for the
  readings: the W sample; for each call, its phase span and route; each reading's cost projected
  **from its phase span** (not its segment count), at each resolution; the backstop you will apply,
  which per the prompt reports a reading past 30 minutes or past its projection and **stops it only
  at a projected day or more, counting the chunks in flight, not only those finished**; and how the
  readings will be grouped into runs. The orchestrator checks it and resumes you. Captures, J0 and
  Part S may run before this stop or after it, as you judge.
- **Launch, one check, end the turn** (`CLAUDE.md`, run registry rule 4). Anything that may exceed
  about ten minutes is registered (`--campaign levin-phase-floor --prompt 06j`) and launched
  detached (`nohup sh -c …`, manifest first, stdout and stderr into the run directory, pid recorded).
  Verify in **one** check that it started, then end your turn and report the run id, the run
  directory, the detached Python pid, the `sh` pid and the first checkpoint or heartbeat line.
  **Do not poll, do not `sleep`-wait, never `pgrep -f`.** The orchestrator waits on the pid and
  resumes you with the final `status.json` (or the exit).
- **Stop and ask** at any of prompt 06j §7's conditions, and at anything this brief or the prompt
  does not settle that changes what you measure. To stop, end your turn with the question stated
  plainly **as its first line**, then the state of every file and run you have made. Do not guess
  and do not repair. Where the prompt's literal wording and the correct measurement part, do what
  is correct and record it as a deviation with both readings; if the two readings would give
  different verdicts, stop and ask.
- **Number and classify every deviation** from the prompt: `STRUCTURALLY REQUIRED`,
  `IMPLEMENTATION CHOICE` or `UNINTENDED DRIFT`. Failed attempts are deviations too.
- **Correctness is the only objective.** An argument from cost or time for a shortcut is a deviation
  to record, not a reason to take it.
- **Formatting:** your copied scripts are records (left as run); do not run `black` over files
  anywhere outside `var/levin-phase-floor-06j/`.

### Your records

In the main checkout's `var/levin-phase-floor-06j/`:
- `log.md`, in campaign README §5.1's form with **every heading of it**, for the laptop to copy to
  `prompts/levin-phase-floor/logs/06j-slow-leaves-and-v2-rows.md`;
- `verification.md`, the text of `docs/levin-phase-floor/VERIFICATION.md` §6j, with links in the
  `06j/…` form (the laptop commits your directory as `docs/levin-phase-floor/06j/`);
- `issues.md`: anything to open on the board, or "none";
- `out/` (with `out/results.md`, which states each verdict with its numbers and what a remedy would
  have to do, and **proposes no code**), `scripts/` (with the diff against 06g's copies) and
  `state/`;
- a provenance header on every output (README §5 rule 5);
- `KEPT_ON_MAC_STUDIO.txt`: every pickle and every file over 2 MB you made, with SHA-256 and size.

When you finish, end with a short report: J0, S1–S4 and W1–W4 with their numbers, every deviation,
every run id, and anything left running (there should be nothing).
