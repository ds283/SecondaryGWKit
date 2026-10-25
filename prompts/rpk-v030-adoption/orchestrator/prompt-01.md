# Orchestrator — prompt 01, switch to the `raypoolkit` rehearsal lock

Read [`../README.md`](../README.md) first: §0–§0.2, §1, §4, §5, §6.2 and §7. This is the
campaign's only note. Its conventions are `dk-v050-adoption`'s
([`../../dk-v050-adoption/orchestrator/prompt-01.md`](../../dk-v050-adoption/orchestrator/prompt-01.md))
and, for the rehearsal, `rrk-v020-adoption`'s 03 (`docs/rrk-v020-adoption-verification.md` §1.1),
carried over. What differs is said here.

**You do not write code.** You may:
- run the suites, `pip` read-only commands, `python -m runregistrykit list` and the lock's `status`
  and `ray-processes`;
- read RayPoolKit's repository with `git`, never changing it;
- wait on the rehearsal's driver pid, once, in the background;
- run a probe from the session scratchpad, never committing one;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`01-switch-to-the-raypoolkit-rehearsal-lock.md`](../01-switch-to-the-raypoolkit-rehearsal-lock.md)
**Board items:** P1, D1, T1, C1, R1, B1 · **Closes:** nothing · **Opens:** nothing expected ·
**Model:** Opus, as the prompt recommends.

**Gate:**
- The campaign is planned (`2b27509`) and gate E holds. The tree is clean.
- `HEAD` is the commit that adds this note, or a later one that touches no code.
- Nothing is `running` or stale in the registry.
- The venv holds `raypoolkit` 0.2.0 at `eaef0ce6…`, `datastorekit` 0.5.0 at `f47b9a25…` and
  `runregistrykit` 0.2.0 at `0bae3a3c…`, none editable.
- Neither lock location exists; the old helper's `ray-processes` lists nothing.
- `git worktree list` shows the main checkout alone.

## 0. What makes this prompt unusual

**The install takes effect before the commit, and the commit does not undo it** (README §4.1). The
review checks the venv directly, not only the diff.

**It runs real Ray under a lock, and the lock's process rule reads command lines.** Contract §6:
a process whose command line names `/tmp/ray-sgk` seeds the walk, and when it is an ancestor of
the lock command, everything else it started is listed. An agent's Bash command is run by a shell
whose own command line is the whole command, so **a lock command must be the only process its
shell command starts while it runs**:
- redirect with `>> file 2>&1` and record `$?` with the builtin `echo`; **never a pipe**, and in
  particular never `| tee`: the `tee` is a sibling under a shell that names the temp dir, so it is
  listed, and `acquire` exits 4 or `ray-processes` exits 1 on a process that is not Ray's;
- never in the same shell command as the launch, the wait or the readers;
- `--purpose` names no temp dir, so neither the `sh -c` of the launch nor its `nohup` names it.

**The checklist is not at the tag.** The prompt's "Read first" 3 says to read both RayPoolKit
documents with `git show v0.3.0:<path>`. The contract is there; the checklist
`docs/adoption/rehearsal-lock/secondarygwkit.md` is not. It was written after the release, in
`127b2b4`, and is read at RayPoolKit's `HEAD`, **`feae4e5`** (clean, `main`, equal to
`origin`). `git diff v0.3.0 feae4e5 -- raypoolkit pyproject.toml docs/rehearsal-lock.md` is empty,
so the contract and the code it describes are the tag's either way. The agent records the reading
as a deviation, `STRUCTURALLY REQUIRED`. README's line 9–13 says the same thing wrongly; that is
the review's residue (§4), not the agent's.

**The guard test's key.** Prompt §2.4 compares "the set of `(file, enclosing function or
<module>)`" with README §0's ten. The three wrapper calls are each inside a nested function named
`_local_ray_init`, and two of them are in one file, so keyed on the innermost function the set has
**nine** members, and a third `_local_ray_init` in `quadsource_atol_sweep.py` would go unseen. The
key must be the **outermost** enclosing function (`run_child`, `run_build`, `main`) or the
qualified path (`run_child._local_ray_init`), compared as a sorted list with its length, so that
ten stays ten and a duplicate fails. That is how README §0 names them; the agent records its form
as an `IMPLEMENTATION CHOICE`.

**What `run_child` does not get.** The rehearsal runs `--build`, which is `run_build`. `run_child`
(the tolerance sweep's per-pair child) gains `address="local"` and the temp dir, checked by `ast`
only: no real-Ray run of it is in scope. The review notes it; it is not a stop.

**What moves on purpose:** `requirements.txt:76`; the venv's `raypoolkit`; `config/ray_local.py`
(new); the ten starts in `main.py`, the six readers and the two drivers; the new test module; the
helper and its tests (deleted); `CLAUDE.md`'s pin, rule 5's Ray check and the lock section;
`measure/out/01-*`, the log and the board; `docs/OPEN_ISSUES.md` only with a §3 entry.

**What must not move:** `config/fingerprint.py`; every existing test module; every golden file;
`datastorekit` and `runregistrykit` in the venv; `docs/` and every other campaign's `prompts/`;
anything under `var/` but the new run, its launch log, the rehearsal store and the readers' output.

**Facts checked by the orchestrator on 2026-10-25 at `2b27509`.** Pass them on.
- Tree clean; worktree list the main checkout alone; nothing `running` or stale (51 runs).
- Neither `~/ScienceData/.rehearsal-lock` nor `~/.raypoolkit/rehearsal-lock` exists; the helper's
  `status` says free and `ray-processes` none. No Ray process is running on the laptop.
- `package_provenance` as the Gate. Ray 2.43.0.
- RayPoolKit: `v0.3.0` peels to `67a75b97a656accb98195a4bda93f64e6a5f3fb2`, locally and by
  `ls-remote`.
- §2.5's grep finds the 18 lines (4 `CLAUDE.md`, 5 helper, 9 tests); `tools.tests.test_rehearsal_lock`
  runs 42 tests. Nothing outside `CLAUDE.md` and the helper's own files names the helper.
- The grep for Ray starts finds the ten calls at README §0.2's lines and nothing else live. Every
  live `.py` under the walk of §2.4 (118 files) parses with `ast`; `.claude/` (worktrees) is
  excluded by the dot rule.
- The rehearsal's inputs exist: `docs/dk-and-rpk-migration/rehearsal/compare_with_recorded.py`,
  `run_readers.sh` and `var/runs/rrk-v020-adoption-03-reference-20261018T232838`. Neither
  `var/datastores/rpk-v030-adoption-01-reference.sqlite` nor `var/rpk-v030-adoption-01/` exists.
  61 GB free.
- Load average about 20 on 10 cores from unrelated desktop processes (Spotlight, sync, Creative
  Cloud), none of them Ray. README §7's four wall-clock tests may fire.

## 1. Before you dispatch

1. The Gate, as §0's facts.
2. **Baselines**, the seven roots serially at `2b27509` (code identical to this note's commit), in
   the session scratchpad: recorded in the board's dispatch record when 01 lands, and compared with
   the agent's §2.1 counts in check 4.
3. **The `var/` marker**, for check 7: `touch` a file in the scratchpad immediately before
   dispatch.

## 2. Dispatch

One fresh-context subagent, on **Opus**, in the main checkout. Give it the prompt, `HEAD`, the
baselines, and §0 of this note up to "## 1. Before you dispatch", as a copy in the session
scratchpad. The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the end**, staged by explicit path from the prompt's §3 acceptance 6.
- **Baselines before the install.**
- **The checklist is read at `feae4e5`**, the contract at `v0.3.0` (§0).
- **The guard's key** is the outermost function, compared with its length (§0).
- **Lock commands: one per shell command, no pipe** (§0).
- **The rehearsal, as R3's was:** launch detached, make the one check of §2.8.3 (lock commands
  each in its own shell command), then **end the turn**, reporting the driver's Python pid, the
  `sh` pid, the run id and the lock's owner file. The orchestrator waits on the pid, once, and
  resumes the agent with the final `status.json`. The agent does not wait or poll.
- **The wall-clock flakes** are re-run alone and recorded, not a stop.
- **Stop and ask** on any of the prompt's §4 conditions; if it stops with the lock held, it leaves
  it held and prints the owner file.

## 3. The review — eight checks

1. **Scope.** `git show --stat <01>`: exactly acceptance 6's paths. `git diff <01>~1 <01> --
   requirements.txt` is the one line. No existing file under any `tests/` changes but the two
   deleted.
2. **The venv, measured here.** `package_provenance`: `raypoolkit` 0.3.0 at `67a75b97…`,
   `requested_revision` `v0.3.0`, not editable; the other two unchanged. `pip check` clean;
   `pip install --dry-run -r requirements.txt` installs nothing. `01-pip.txt` has the dry run before
   the install, naming `raypoolkit-0.3.0` alone.
3. **The code.** `config/ray_local.py`'s value is `"/tmp/ray-sgk"`, spelled once. Each of the seven
   scripts' `ray.init` is the form of prompt §2.3, a top-level expression statement; the three
   wrappers pass `_temp_dir=RAY_TEMP_DIR` and `address="local"`; nothing added to `run_child`
   contains lower-case `rehearsal`. `main.py`'s comment gains one sentence. The §2.5 grep at
   `<01>` finds one line, `CLAUDE.md`'s on the old location. `black --check` clean on every touched
   file.
4. **The roots, re-run here at `<01>`.** `tools` = baseline − 42, `RunDrivers` = baseline + the
   new tests, the rest equal, all OK, a known flake re-run alone.
5. **The guard.** Read the test: it walks as prompt §2.4 says, keys as §0 says, and its docstrings
   name the failure. The log's M1–M5, each applied as a diff, each failing a named test, each
   reverted (`git status` clean of them). Re-apply **M4** here in the scratchpad's copy of the
   tree, or by hand and revert, if the log's record of it is thin.
6. **The rehearsal and the lock.** From `01-lock.txt` and `01-rehearsal.txt`: `acquire` 0;
   `set-pid` 0 with the Python pid; `ray-processes` 1 listing `gcs_server` and `raylet` under
   `/tmp/ray-sgk`; `release` 1 while it ran; second `acquire` 3; `status` HELD. The run `done`,
   exit 0, `raypoolkit` 0.3.0 in its manifest's `packages`; the fingerprint **`39809dca…`**,
   recomputed here with `store fingerprint` (no `--write`); the six readers 0 with the store
   unchanged; `ray-processes` 0 after; `release` 0; `status` free with no `old location` line.
   Neither lock location exists now, and no process names `/tmp/ray-sgk`.
7. **Nothing else moved.** `find var -newer <marker>` lists only the run directory, its launch
   log, the rehearsal store and its sidecar, and `var/rpk-v030-adoption-01/`. `git status --short`
   clean. RayPoolKit unchanged at `feae4e5`, clean.
8. **The records.** `CLAUDE.md`'s diff against prompt §2.6: the pin clause, rule 5's command, and
   the lock section stating the protocol with full `acquire`, `set-pid` and `release` lines,
   exit 3 and the old location, the temp-dir rule, one command per shell command, no `auto` and no
   `ray stop`; the brief/§4.1 bullet gone; no longer than now by more than a few lines. The log has
   every README §5.1 heading and classifies each deviation (the checklist's revision and the guard's
   key among them). The board as prompt §6. The commit message in `CLAUDE.md`'s form, ending
   `Co-Authored-By: Claude Opus …`.

## 4. After the review

If it passes, residue first, in a follow-up commit of its own:
- "this commit" → the SHA, on the board and in the log;
- README's opening (lines 9–13) and the prompt's "Read first" 3: the checklist is at RayPoolKit
  `127b2b4`, after the tag, read at `feae4e5` — a correction of the plan's own text, recorded as
  such;
- README §2's 01 row status.

Then **the close**, one commit of the orchestrator's: the board's header (**CLOSED — 1 of 1 landed
and reviewed**), its §1 row, gate FP, `### 5.2 Review and close (<date>)` holding what each check
found, and the campaign's line in `prompts/INDEX.md` ("closed 1 / 1, <date>") with the index's
header count. Anything that is not residue is opened on the board's §3 and in the index instead.

**Report to the user:** the pin and the venv; the lock seen holding and refusing under real Ray;
the fingerprint; the roots; that the rehearsal store is theirs to retire (README §3); that
`handover` 09 is unblocked; that ChamPBH and StochasticInstantons still take the old location, and
SGK's `acquire` exits 3 while either holds it (U2).
