# Orchestrator — prompt 01, pin `runregistrykit` at `v0.2.0`

Read [`../README.md`](../README.md) first: §0, §0.2, §2.2 (01's charter), §4, §4.2, §5 (rules 1–6,
9–14), §6.2 (C4) and §8. Then the board's §2 (gate E). This is the campaign's first note, so it
states the conventions in full. They are `runregistry-extraction`'s
([`../../runregistry-extraction/orchestrator/prompt-01.md`](../../runregistry-extraction/orchestrator/prompt-01.md)
§0's last list), carried over, and that campaign's
[`prompt-05.md`](../../runregistry-extraction/orchestrator/prompt-05.md) is this note's model: 01 is
its 05 again, for the next tag. What differs is said here.

**You do not write code.** You may:
- run the suites, `pip` read-only commands, and `python -m runregistrykit list` and `store show`;
- read RunRegistryKit's repository with `git`, never changing it;
- run a probe or comparison script from the session scratchpad, never committing one;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`01-pin-runregistrykit-at-v0.2.0.md`](../01-pin-runregistrykit-at-v0.2.0.md)
**Board items:** K1, B1 · **Closes:** nothing · **Narrows:** nothing · **Opens:** nothing
expected · **Model:** Sonnet, as the prompt recommends.

**Gate:**
- The campaign is planned (`8a8cd10`) and gate E holds. The tree is clean.
- `HEAD` is the commit that adds this note, or a later one that touches no code (outside `prompts/`
  and `docs/`).
- Nothing is `running` or stale in the registry; the rehearsal lock is free; `ray-processes` lists
  nothing.
- The venv holds `runregistrykit` 0.1.0 at `9f8ebe43…`, not editable.
- `git worktree list` shows the main checkout alone. One prompt at a time in this checkout
  (README §4.2).

## 0. What makes this prompt unusual

**The install takes effect before the commit, and the commit does not undo it.** Unlike
`runregistry-extraction` 05, the package is already imported by every driver, `list` and `store`
command, so from the moment of the install every registry read in this checkout is `v0.2.0`'s.
`git revert` of 01 leaves the venv on `v0.2.0` (README §4.2 gives the reinstall). So the review
checks the venv directly, not only the diff, and the baselines must be taken **before** the
install: they cannot be re-taken under `v0.1.0` afterwards.

**What moves on purpose:**
- `requirements.txt:81`, one line, `@v0.1.0` → `@v0.2.0`. No other line.
- `CLAUDE.md:56` and `:88`, the pin's two mentions, and the one clause §2.5 adds after `:56-57`'s
  sentence. No other line.
- The venv: `runregistrykit` 0.1.0 at `9f8ebe43…` is replaced by 0.2.0 at `0bae3a3c…`. Nothing else
  is installed, upgraded or removed.
- `prompts/rrk-v020-adoption/measure/out/` is created, with `01-pip.txt`, `01-list.txt`,
  `01-store-show.txt` and `01-roots.txt`.
- `logs/01-pin-runregistrykit-at-v0.2.0.md` is created (`logs/` is empty).
- The board: header, §1 row 01, K1 and B1, the G1 row of §2, `### 5.1 After prompt 01`.
- No issue opens or closes. The index stays at 100.

**A prompt claim that the package contradicts, which the agent records and does not chase.**
Prompt §2.4 says the new summary clause's *N* "must count every run directory that has a
`status.json`", which is 43. **`v0.2.0` will print 45.** `list` counts
`sum(1 for entry in entries if not entry["host_recorded"])` (`runregistrykit/__main__.py:659` at
the tag), and `list_runs` sets `host_recorded` to `isinstance(status.get("host"), dict)`
(`__init__.py:1243`), which is false for a directory with no status; its docstring says so
("or a directory with no readable status", `:1198-1199`). The two `a3-pilot` directories have no
`status.json` and no manifest, so they count. **The expected summary line is
`45 run(s) under …/var/runs: 43 finished, 2 unknown; 45 with no host recorded.`** That is
acceptance 4's "count accounted for": 43 statuses written under `v0.1.0`, none with a `host`, plus
the pair with no status. The agent records the prompt's wording as a deviation, `STRUCTURALLY
REQUIRED`, and it is **not a stop**. Any other *N* is.

**What the pair reads as under `v0.2.0`.** No status and no manifest: `liveness` `unknown`
(`__init__.py:636-637`), no `!!` (`_unsaid` requires a manifest, `__main__.py:86`), and no stderr
line. With nothing stale and nothing foreign, `list` prints **nothing on stderr** and exits 0.
Anything on stderr is recorded and is a stop if it names a run.

**The facts, checked by the orchestrator on 2026-10-18 at `8a8cd10`** (`git diff --stat 79434bf
8a8cd10 -- . ':!prompts' ':!docs'` is empty). Pass them on.
- **The release.** Tag `v0.2.0`, object `895e3c7013ae4887011d435f3e800b8f1697a1de`, commit
  `0bae3a3c501a8264ee4be5bc24f06422af38720f`, the same locally and by `git ls-remote origin`.
  Nothing under `runregistrykit/` or in `pyproject.toml` changed from the tag to RunRegistryKit's
  `HEAD`, `f6d4b80`, clean. `pyproject.toml` at the tag: `version = "0.2.0"`,
  `requires-python = ">=3.12"`, `dependencies = []`, `[stores]` = `datastorekit>=0.4.0`,
  `runregistrykit.tests` excluded. The modules are `__init__`, `__main__`, `client`, `fingerprint`
  and `stores`, as at `v0.1.0`.
- **The dry run.** With line 81 moved in a scratch copy of `requirements.txt` (the diff was that
  line alone), `pip install --dry-run -r` printed `Would install runregistrykit-0.2.0` and nothing
  else. It prints `Resolved https://github.com/ds283/RunRegistryKit to commit 895e3c70…`: **that is
  the tag object, not a mismatch**, as `runregistry-extraction` 05's log found for `v0.1.0`
  (`fd5ce48…`). The installed `direct_url.json` records the commit, `0bae3a3c…`. The real install
  will also print an uninstall of `runregistrykit-0.1.0`; that is the replacement, not a removal.
- **`this_host()` at the tag**, run from an extract of the tag in the scratchpad, never the venv:
  `id` `8d2b247824882486`, `name` `mps026088.phys.susx.ac.uk`, `boot`
  `376461AF-4985-4F42-8E41-F4E914F5E3E6`, `machine` `arm64`, `system` `Darwin`, `python`
  `3.12.15`. `TERMINAL_STATES` `('done', 'failed', 'killed', 'lost')`. The agent's reading must
  give the same `id`, `name` and, unless the machine has rebooted, `boot`.
- **The venv now.** Python 3.12.15. `pip freeze`: `runregistrykit` at
  `9f8ebe431503fb8c79ee1407cc460501f1ca2d92`, `datastorekit` at
  `c8570e849334d09085bbd84876095fc9e22bd005`, `raypoolkit` at
  `eaef0ce60e2fc354bb32f0ad576010e6a83c1976`, `ray==2.43.0`, `SQLAlchemy==2.0.39`. None editable.
  `requirements.txt`: `datastorekit` `:17`, `ray` `:75`, `raypoolkit` `:76`, `runregistrykit`
  `:81`, `SQLAlchemy` `:88` — the four lines the adoption note's §1 cites hold.
- **The registry.** `list` exits 0: **45 runs, 43 finished and 2 `unknown`**, nothing `running` or
  stale. Of the 45 directories, 43 have both `status.json` and `manifest.json` (24 `done`, 16
  `killed`, 3 `failed`; **none has a `host` key**); `a3-pilot` and `a3-pilot-resume` have neither.
- **The 21 sidecars**, by `find var/datastores -name '*.manifest.json' | sort`, the primary being
  the sidecar's path with `.manifest.json` replaced by `.sqlite`: 20 at the top of
  `var/datastores/` (`a3-v2-readiness-07-{reference,resumed}`, `datastore-generic-09-rehearsal`,
  `datastore-integrity-10-{reference,resumed}`, `dk-and-rpk-migration-{02,05,08}-{reference,resumed}`,
  `handover-A3-baseline-lambdacdm`, `handover-atol-sweep`, `handover-stale-run-probe`,
  `runregistry-extraction-{02,04,08}-{reference,resumed}`) and
  `backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm`. No `.sqlite` file exists.
  `store show` is read-only at both tags (`__main__.py:7`); its exit status is part of the reading,
  recorded at the foot of each block.
- **The lock and Ray.** Lock free; `ray-processes` none.
- **The roots**, at `8a8cd10`, serially, in the foreground: AdaptiveLevin 32, CosmologyModels 39,
  ObjectFactories 279, tools 91, LiouvilleGreen 148 (1 skipped), RunDrivers 99, ComputeTargets
  700, all OK, no known flake fired (load about 6). Every count is §0.2's.
- **The index:** 100 open; this campaign's one at §1.22.
- **Not touched, and staying:** every code file and test, every `docs/` script, the other pins,
  `README.md`, `prompts/INDEX.md`, RunRegistryKit's checkout, and everything under `var/`.

**What the review exists to establish.**
- **(P1) The pin.** One line moved, to the tag, with the extra; the venv at the tag's commit, not
  editable; nothing else installed, upgraded or removed.
- **(P2) The baselines.** Taken before the install, with the `HEAD` they were taken at; each the
  command's output, whole.
- **(P3) Nothing SGK reads changed but the summary clause.** `list` row for row and `store show`
  block for block equal before and after; the clause's 45 accounted for; importing the package
  loads none of `ray`, `sqlalchemy`, `datastorekit`; the roots at §0.2's counts before and after.
- **(P4) `CLAUDE.md`.** The two pin mentions and the one added clause, and no other change.
- **(P5) The records** in the campaign's form.

**Conventions.** All of these bind the agent, and the brief says so.
- **Staging and reporting.** Stage by explicit path only, and show `git diff --cached --name-only`
  before committing. Count, then paste. A log states only what the diff and the commands show.
- **The SHA.** The agent writes "this commit" wherever 01's hash belongs (the board's §1 row, the
  G1 row, the log), and the orchestrator substitutes it in its follow-up.
- **Generated, not typed.** The four `measure/out/` files are commands' output, redirected or
  pasted whole, each block headed by its command and the `HEAD`. Long runs of
  `Requirement already satisfied` lines may be omitted from the pip output if the omission is
  stated in place, as `runregistry-extraction`'s `05-pip.txt` did; every other line is kept,
  including each `Resolved … to commit` line.
- **Suites.**
  - Use README §8's command, write each root's full output to a file in the agent's scratch
    directory, and put the grep of it into `01-roots.txt`, with the load average.
  - Run each root **in the foreground**, serially, within the turn. A Bash call is capped at ten
    minutes; ComputeTargets takes about two and a half.
  - README §8's four known flakes are re-run alone before being called real, and both runs are
    recorded. A flake is not a stop.
- **Scratch.** The agent uses its own subdirectory of the session scratchpad, never `/tmp` and
  never `var/`.
- **The venv.** Exactly the prompt's two `pip install` commands, `--dry-run` then the real one,
  from the repository root against `requirements.txt` as edited by §2.2. No other `pip install` or
  `pip uninstall`. If the dry run shows anything but `runregistrykit`, it stops before the real
  install.
- **`var/` is read, never written.** `list` and `store show` only. No `store create`, `adopt`,
  `copy`, `move`, `fingerprint`, `retire` or `amend`; no `end`; no `begin()`; no `Run.finish`.
- **Ray.** None. The lock is not taken.
- **Stop conditions are real.** A stop goes to the user through the orchestrator, and the agent
  resumes with `SendMessage`. **If it stops after the install**, it says so plainly and does not
  reinstall `v0.1.0`.
- **Other sessions** may land commits on this branch. The agent checks `git log` before measuring
  and before committing; the review is against the agent's commit.

## 1. Before you dispatch

Done at `8a8cd10` on 2026-10-18, before this note was committed; the results are §0's facts.
1. **The tree.** Branch `handover-remedial`; `git status --short` empty; `git worktree list` shows
   this checkout alone; nothing since `79434bf` touches code.
2. **The registry.** Nothing `running` or stale.
3. **The lock and Ray.** `tools.rehearsal_lock status`: free; `ray-processes`: none.
4. **The venv and the release.** As §0.
5. **Baselines.** The seven roots, serially, at `8a8cd10`:

   | Root | Ran | Result | Wall |
   |---|---|---|---|
   | `AdaptiveLevin/tests` | 32 | OK | 0 s |
   | `CosmologyModels/tests` | 39 | OK | 3 s |
   | `ObjectFactories/tests` | 279 | OK | 17 s |
   | `tools/tests` | 91 | OK | 12 s |
   | `LiouvilleGreen/tests` | 148 | OK (skipped=1) | 28 s |
   | `RunDrivers/tests` | 99 | OK | 4 s |
   | `ComputeTargets/tests` | 700 | OK | 168 s |

   Every count is §0.2's. Load average about 5.5–6.5 on 10 cores; no known flake fired.

6. **The `var/` marker**, for check 6: immediately before dispatch, `touch` a marker file in the
   orchestrator's scratch directory and record its path.

## 2. Dispatch

One fresh-context subagent, on **Sonnet**. Give it the prompt, the README and the board, `HEAD`,
the baselines, the index count, and §0 of this note up to "## 1. Before you dispatch", as a copy in
the session scratchpad. The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the end**, staged by explicit path from the prompt's §6 list. `measure/out/`
  does not exist yet; it creates it.
- **Baselines before the install**, all four, and the agent saves the `HEAD` with each.
- **The two `pip install` commands, and no other venv change.** The tag object in the dry run and
  the uninstall of 0.1.0 in the real install are expected (§0).
- **The summary clause will say 45**, not 43 (§0): recorded as a deviation, not a stop.
- **The wall-clock flakes** are expected, re-run alone and recorded, not a stop.
- **Stop and ask** on any of the prompt's §4 conditions; if it stops after the install, say so.

## 3. The review — seven checks

1. **Scope.** `git show --stat <01's commit>`: `requirements.txt`, `CLAUDE.md`, the four
   `measure/out/01-*` files, the log and the board; `docs/OPEN_ISSUES.md` only with a new §3 entry,
   with its count and date. `git diff <01>~1 <01> -- requirements.txt` is the one changed line.
   Nothing under `README.md`, `prompts/INDEX.md`, `orchestrator/`, any code, any test, or `docs/`
   other than the index.
2. **P4, `CLAUDE.md`.** `git diff <01>~1 <01> -- CLAUDE.md`: `:56`'s `v0.1.0` → `v0.2.0`, the
   clause "and moved to `v0.2.0` on 2026-10-18 in the `rrk-v020-adoption` campaign" after
   "adopted in the `runregistry-extraction` campaign", and `:88`'s `v0.1.0` → `v0.2.0`. Paragraphs
   re-wrapped at ~100 columns are acceptable; a changed word elsewhere is not.
   `grep -n 'v0\.1\.0' CLAUDE.md` finds nothing.
3. **P1, the venv, measured here.** `pip show runregistrykit` and its `direct_url.json`: 0.2.0,
   `commit_id` `0bae3a3c…`, `requested_revision` `v0.2.0`, no `dir_info`; `pip show -f` lists no
   `runregistrykit/tests/`. `pip freeze` against the agent's before-freeze differs by the one line;
   `datastorekit` and `raypoolkit` unchanged; `pip check` clean; `pip install --dry-run -r
   requirements.txt` now installs nothing. `01-pip.txt` has the dry run before the install.
4. **P3, the imports and the host.** The `sys.modules` check re-run here prints `[]`.
   `package_provenance(...)` equals the agent's after-reading, and its `datastorekit` and
   `raypoolkit` entries the before-reading. `TERMINAL_STATES` ends `lost`; `this_host()` matches
   §0's `id` and `name`, and the agent's `boot`.
5. **P2 and P3, the readings.**
   - `01-list.txt`: both readings, each with its `HEAD`, exit status and stderr. Parse both into
     `(id, marker, state)` rows with a scratch script and compare: 45 equal rows. The after-summary
     is §0's line exactly. A fresh `list` here gives the same rows.
   - `01-store-show.txt`: 21 blocks before and 21 after, each headed by its command and ending with
     its exit status, the backup's among them. Diff the two halves block for block: equal. A fresh
     `store show` on three of them, the backup's among them, equals the after-blocks.
   - `01-roots.txt`: seven roots before and after, at §0.2's counts; a known flake re-run alone
     shows both runs. Re-run the seven here at the agent's commit, as §1's table.
6. **Nothing else moved.** `find var -newer <marker>` lists nothing, or only files written by a
   process that is not the agent's (named, if any). `git status --short` is clean. RunRegistryKit's
   checkout is unchanged, at `f6d4b80`, clean.
7. **P5, the records.** The board: header (status and date), §1 row 01 (✅, "this commit", the
   log), K1 and B1 ✅, G1's row as the prompt words it, `### 5.1 After prompt 01 (2026-10-18)`.
   The log has every §5.1 heading, classifies each deviation (the summary clause among them), says
   "none" under deliberate breakage, and records §2.6's check of the adoption note's §1 and §2.
   The commit message is in `CLAUDE.md`'s form, subject imperative and under ~72 characters,
   ending `Co-Authored-By: Claude Sonnet …`.

## 4. After the review

- **Record it on the board** as an *Orchestrator review of prompt 01* paragraph in §5.1: what each
  check found, and anything handed on to 02. Fix small residue in a follow-up commit of its own:
  - "this commit" → the SHA, on the board and in the log;
  - the board's line on orchestrator notes, listing this one as "used for 01";
  - `prompts/INDEX.md`'s row for this campaign: 1 of 4 landed;
  - README §2's 01 row status, and §2.2 01's acceptance 3 wording on the summary clause, if the
    agent's measurement confirms §0 (a correction of the plan's own text, recorded as such).
- Anything that is not residue is opened on the board's §3 and in the index.
- **Report to the user:** the pin and the venv; the import check and `this_host()`; the readings
  before and after; the roots; that RunRegistryKit's G1 now holds on SGK's side and its record is
  the user's (README §6.3); what 02 inherits.
- **Then 02**, the adoption. Its note is written after this review.
