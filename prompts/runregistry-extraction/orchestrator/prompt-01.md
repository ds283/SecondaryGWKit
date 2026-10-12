# Orchestrator — prompt 01, freeze the run registry

Read [`../README.md`](../README.md) first: §0, §0.2, §2.2 (01's charter), §4, §5 (rules 1–6, 9, 10,
12 and 14), §6.2 (C4, C7) and §8. Then [`../runregistrykit-extraction-brief.md`](../runregistrykit-extraction-brief.md)
§2, which is what the frozen tree is for. This is the campaign's first note, so it states the
conventions in full. They are `dk-and-rpk-migration`'s
([`../../dk-and-rpk-migration/orchestrator/prompt-01.md`](../../dk-and-rpk-migration/orchestrator/prompt-01.md)
§0's last list), carried over and cut to a prompt that adds no code.

**You do not write code.** You may:
- run the suites, `python -m RunRegistry list` and `store show`, and `git ls-tree`;
- run a comparison script from the session scratchpad, never committing one;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`01-freeze-the-run-registry.md`](../01-freeze-the-run-registry.md)
**Board items:** F1–F3 · **Closes:** nothing · **Narrows:** nothing · **Opens:** nothing
expected · **Model:** Sonnet, as the prompt recommends.

**Gate:**
- The campaign is planned (`526036c`). The tree is clean.
- `HEAD` is the commit that adds this note, or a later one that touches no code.
  - "Code" here means anything outside `prompts/` and `docs/`, and in particular `RunRegistry/`.
  - If a later commit touches `RunRegistry/`, the freeze's acceptance 1 fails as written: stop and
    ask the user before dispatching.
- Nothing is `running` in the registry, and no Ray process of this repository is up.
- One prompt at a time in this checkout (README §4.2).

## 0. What makes this prompt unusual

**It adds no code and changes nothing it measures.** One section of `CLAUDE.md`, one table, three
saved readings. So the review has three halves:
- **the freeze is stated where every later session reads it**, in the words the prompt asks for,
  and nothing else in `CLAUDE.md` moves;
- **the frozen tree is exact**: 25 rows, every blob hash `git ls-tree`'s, because E2's G1 compares
  RunRegistryKit's import against this table and nothing else;
- **the baselines are true and were taken read-only**: nothing under `var/` and nothing under
  `RunRegistry/` changes.

**The table is the deliverable another repository depends on.** RunRegistryKit's campaign will check
its import file by file against §2.1 (brief §2). A hash copied by hand, truncated, or taken from the
working tree with `git hash-object` on a file with local changes would make G1 unverifiable. It is
generated from `git ls-tree -r HEAD RunRegistry/`, not typed.

**What the freeze covers, and what it does not.** It covers every tracked file under `RunRegistry/`,
its tests included. It does **not** cover the callers: the two `docs/` drivers, the rehearsal
tooling, `ObjectFactories/tests/test_quadsource_policy_key.py` and the rest of audit §2.1's list stay
changeable (06 rewrites them). The `CLAUDE.md` section must not say or imply otherwise.

**A record the commit makes false, which the prompt's file list allows the agent to correct**
(README §6.2 C7): the board header's **Status** line ("none landed"). The agent corrects it and
logs the correction as `IMPLEMENTATION CHOICE`, citing C7 and this note.
`prompts/INDEX.md`'s row also says "none landed", but INDEX.md is not on the prompt's list: the
orchestrator corrects it in its follow-up (§4). README §2's "written 2026-10-12" stays true and is
not touched.

**What moves on purpose:**
- `CLAUDE.md` gains one section between "Long-running jobs — the run registry" (`:77-104`) and
  "Repository mechanics" (`:106`). No other line of it moves.
- `prompts/runregistry-extraction/measure/out/` is created, with `01-list.txt`,
  `01-store-show.txt` and `01-roots.txt`.
- `prompts/runregistry-extraction/logs/01-freeze-the-run-registry.md` is created (`logs/` does not
  exist yet).
- The board: the header, the §1 row for 01, F1–F3, §2.1's table and `### 5.1 After prompt 01`.
- No issue opens or closes. The index stays at 105.

**The facts, checked by the orchestrator on 2026-10-12 at `e3a7077`** (`git diff --stat c8a7f14
e3a7077 -- . ':!prompts' ':!docs'` is empty; the three commits since the audit,
`c522c7a`, `526036c` and `e3a7077`, touch only `docs/` prose and `prompts/`). Pass them on.
- **The frozen tree.** `git ls-tree -r HEAD RunRegistry/` gives **25** files, as does
  `--name-only | wc -l`. `git diff --stat c8a7f14 HEAD -- RunRegistry/` is empty.
- **The registry.** `python -m RunRegistry list` shows 29 runs: 27 finished (`done`, `killed` and
  `failed`, all `finished` by liveness) and 2 `unknown` (`a3-pilot`, `a3-pilot-resume`), none
  `running` or stale.
- **The 14 sidecars**, by `find var/datastores -name '*.manifest.json' | sort`, the primary being
  the sidecar's path with `.manifest.json` replaced by `.sqlite`:
  - `a3-v2-readiness-07-{reference,resumed}`, `datastore-generic-09-rehearsal`,
    `datastore-integrity-10-{reference,resumed}`, `dk-and-rpk-migration-{02,05,08}-{reference,resumed}`,
    `handover-A3-baseline-lambdacdm`, `handover-atol-sweep`, all under `var/datastores/`;
  - and `var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm`.

  `store show` on each, from the repository root with `PYTHONPATH=.`, gives `kind: registry`,
  `problems: none`, a retirement block with `state: retired`, and exit status 0. `store show`
  returns 1 for a retired store with problems (`RunRegistry/__main__.py:180-182`), so **the exit
  status is part of the reading**: the agent records it at the foot of each block.
- **Read-only.** `list` and `store show` only read (`__main__.py:136-182`, `:476-512`). The
  orchestrator ran all fourteen and `find var -newer <marker>` was empty afterwards.
- **The roots**, run by the orchestrator at `e3a7077` before dispatch, serially, in the foreground,
  load average about 8 on 10 cores (another session's Python process, not Ray):

  | Root | Ran | Result | Wall |
  |---|---|---|---|
  | `AdaptiveLevin/tests` | 32 | OK | 1 s |
  | `CosmologyModels/tests` | 39 | OK | 2 s |
  | `ObjectFactories/tests` | 271 | OK | 15 s |
  | `RunRegistry/tests` | 286 | OK | 91 s |
  | `tools/tests` | 49 | OK | 12 s |
  | `LiouvilleGreen/tests` | 148 | OK (skipped=1) | 30 s |
  | `ComputeTargets/tests` | 693 | OK | 140 s |

  Every count is §0.2's. No known flake fired.
- **The venv.** `datastorekit` 0.2.2 (`6d8ee9ae…`, `v0.2.2`) and `raypoolkit` 0.2.0
  (`eaef0ce6…`, `v0.2.0`), both VCS installs, neither editable. `runregistrykit` is not installed.
- **The index:** 105 open; this campaign's two at §1.21.
- **Not touched, and staying:** every file under `RunRegistry/`, every test, every `docs/` script,
  `requirements.txt`, the venv, and every store and run directory under `var/`.

**What the review exists to establish.**
- **(E1) The freeze section.** In place, titled as the prompt says, making its four points, claiming
  nothing beyond them; the rest of `CLAUDE.md` byte-identical.
- **(E2) The frozen tree.** §2.1 equals `git ls-tree -r <01's commit> RunRegistry/` row for row, and
  equals `c8a7f14`'s.
- **(E3) The baselines.** Each file is what its command prints, with the `HEAD` it was taken at,
  and agrees with the orchestrator's reading above.
- **(E4) Nothing it measures moved.** `RunRegistry/` and `var/` unchanged by the agent's session.
- **(E5) The records.** The board, the log and the commit are in the campaign's form.

**Conventions.** All of these bind the agent, and the brief says so.
- **Staging and reporting.** Stage by explicit path only, and show `git diff --cached --name-only`
  before committing. Count, then paste. A log states only what the diff and the commands show.
- **The SHA.** The agent writes "this commit" wherever 01's hash belongs (the board's §1 row, §2.1's
  lead-in, the log), and the orchestrator substitutes it in its follow-up.
- **Generated, not typed.** §2.1's rows and the three `measure/out/` files are produced by commands
  and pasted or redirected whole. The agent may reformat `git ls-tree`'s output into the table's
  two columns with a one-line `awk` or Python, but does not edit a hash.
- **Suites.**
  - Use README §8's command, write each root's full output to a file in the agent's scratch
    directory, and put the grep of it into `01-roots.txt`.
  - Run each root **in the foreground**, serially, within the turn. A Bash call is capped at ten
    minutes; ComputeTargets takes about two and a half.
  - Under load, four tests are known to fail on wall clock or chance (README §8):
    - `test_tk_wkb_phase.TestCost.test_wall_time_per_object`;
    - `test_background_tau.test_build_cost`;
    - `test_numeric_phase_cut.test_the_print_is_gated_both_ways`;
    - LiouvilleGreen's `test_3bessel_analytic.test_JJJ`.

    Re-run a failure alone before calling it real, and record both runs in `01-roots.txt`.
- **Scratch.** The agent uses its own subdirectory of the session scratchpad, never `/tmp` and
  never `var/`.
- **The venv.** The agent installs nothing, uninstalls nothing and runs no `pip install`.
- **Ray.** The agent starts no Ray and runs no build (02 does).
- **`var/` is read, never written.** No `store create`, `adopt`, `copy`, `move`, `fingerprint`,
  `retire` or `amend`; no `begin()`.
- **Stop conditions are real.** A stop goes to the user through the orchestrator, and the agent
  resumes with `SendMessage`.
- **Other sessions** may land commits on this branch. The agent checks `git log` before measuring
  and before committing; the review is against the agent's commit.

## 1. Before you dispatch

Done at `e3a7077` on 2026-10-12, before this note was committed; the results are §0's facts.
1. **The tree.** Branch `handover-remedial`; `git status --short` empty; `git worktree list` shows
   this checkout alone; nothing since `c8a7f14` touches code.
2. **The registry.** Nothing `running` or stale.
3. **Ray.** `pgrep -lf 'gcs_server|raylet|ray::'` finds nothing.
4. **The venv.** As §0.
5. **Baselines.** The seven roots, as §0's table.
6. **The `var/` marker**, for check 6: immediately before dispatch, `touch` a marker file in the
   orchestrator's scratch directory and record its path.

## 2. Dispatch

Dispatch one fresh-context subagent, on **Sonnet**. Give it:
- the prompt;
- the campaign README and this board;
- `HEAD`, the baselines and the index count;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad.
  The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the end.** Its log is `logs/01-freeze-the-run-registry.md`, in README §5.1's
  form. `logs/` and `measure/out/` do not exist yet; it creates them.
- **The files it may add or change are the prompt's §6 list, and nothing else:** `CLAUDE.md`; the
  three files under `measure/out/`; the log; this campaign's board; and `docs/OPEN_ISSUES.md` only
  if §3 gains an entry. In particular, not `README.md`, `prompts/INDEX.md`, `orchestrator/`,
  anything under `RunRegistry/`, or `var/`.
- **The freeze covers `RunRegistry/` only**, not its callers (§0).
- **The board's header** is corrected as §0 says.
- **The deliberate-breakage record** says none, by acceptance 6, since the prompt adds no code or
  test.
- **Stop and ask** on any of the prompt's §4 conditions.

## 3. The review — seven checks

1. **Scope.** `git show --stat <01's commit>` touches only the prompt's §6 list.
   - The diff over `RunRegistry/`, `var/`-tracked paths (none), `README.md`, `prompts/INDEX.md` and
     `orchestrator/` is empty.
   - `docs/OPEN_ISSUES.md` is untouched, unless the agent opened an issue, in which case the row,
     the count and the date move in the same commit.
2. **E1, by reading.** Read the new section in full.
   - It sits between "Long-running jobs — the run registry" and "Repository mechanics", with the
     prompt's title exactly.
   - It says: frozen since prompt 01 of `runregistry-extraction`, because RunRegistryKit imports it
     and checks the import against the frozen tree; nothing under `RunRegistry/`, tests included,
     changes until 06 replaces it with the pinned `runregistrykit`; a defect is recorded on the
     campaign's board and in `docs/OPEN_ISSUES.md` and fixed in RunRegistryKit, and if a fix cannot
     wait the session stops and asks; the six rules above still bind.
   - It does not freeze the callers, name a hash it could not know, or restate the six rules.
   - It is "a few lines", in the register of the sections around it.
   - `git diff <01's commit>~1 <01's commit> -- CLAUDE.md` is a single inserted hunk: no other line
     of `CLAUDE.md` changes.
3. **E2, by script.** In the scratchpad, regenerate `git ls-tree -r <01's commit> RunRegistry/`
   and parse §2.1's table from the committed board.
   - 25 rows, path order, every hash 40 hex, and the two sets of `(path, blob)` pairs equal.
   - Both equal `git ls-tree -r c8a7f14 RunRegistry/` (saved at
     `scratchpad/orch01/ls-tree-e3a7077.txt`, which equals `c8a7f14`'s).
   - The row count the board states equals 25 and the log says how it was compared.
4. **E3, `01-list.txt`.** Its `HEAD` is recorded; it is the command's output verbatim; its summary
   line reads 29 runs, 27 finished and 2 `unknown`; its run ids and states equal a fresh `list` at
   review time, row for row (ages aside).
5. **E3, `01-store-show.txt` and `01-roots.txt`.**
   - Fourteen blocks, each headed by its command, the backup's among them; each block shows
     `kind: registry`, `problems: none`, `state: retired` and exit status 0. Compare each block's
     `fields` against a fresh `store show` at review time: equal.
   - `01-roots.txt`: seven roots, each with its `Ran` and `OK`/`FAILED` lines, counts as §0's
     table. A known flake re-run alone shows both runs. The `HEAD` and the load average are
     recorded.
6. **E4, nothing moved.** `find var -newer <marker>` is empty. `git diff --stat c8a7f14 <01's
   commit> -- RunRegistry/` is empty. `git status --short` is clean after the agent's commit.
7. **E5, the records.**
   - The board: the header's status and date; the §1 row (✅, "this commit", the log); F1–F3 ✅;
     §2.1; `### 5.1 After prompt 01 (2026-10-12)`.
   - The log has every §5.1 heading, classifies each deviation, and its counts agree with the three
     files.
   - The commit message is in `CLAUDE.md`'s form, subject imperative and under ~72 characters,
     ending `Co-Authored-By: Claude Sonnet …`.

## 4. After the review

- **Record it on this board** as a §5.1 *Orchestrator review of prompt 01* paragraph: what each
  check found, and anything handed on to 02. Fix small residue in a follow-up commit of its own.
  That includes:
  - "this commit" → the SHA, on the board and in the log, and the SHA added beside "prompt 01 of
    `runregistry-extraction`" in `CLAUDE.md`'s section, so that a later session can name the frozen
    commit without searching;
  - the board's line on orchestrator notes, listing this one as "used for 01";
  - `prompts/INDEX.md`'s row: 1 of 9 landed.
- Anything the review finds that is not residue is opened on this board's §3 and in the index.
- **Report to the user:**
  - the freeze section, quoted;
  - the frozen commit and the 25-row table's agreement with `git ls-tree`;
  - the three baselines in one line each;
  - that `RunRegistry/` and `var/` are unchanged;
  - that the brief's import commit is now named, for the user to pass to RunRegistryKit's session.
- **Then 02**, R0. Its note is written after this review.
