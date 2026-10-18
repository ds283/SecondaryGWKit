# Log — rrk-v020-adoption prompt 01, pin `runregistrykit` at `v0.2.0`

- **Subject:** [`01-pin-runregistrykit-at-v0.2.0.md`](../01-pin-runregistrykit-at-v0.2.0.md), board items K1 and B1.
- **Commit:** `0a27b0c` (parent `f0d280a`).
- **Result:** landed. K1 and B1 pass; no stop condition of the prompt's §4 fired. All seven roots pass at §0.2's counts
  before and after, and no known flake fired, so none was re-run.
- **Model:** Sonnet 5.5.
- **Date:** 2026-10-18.

The orchestrator's facts for this prompt (its note's §0) were read first and are cited as "the note".

## What shipped

- **`requirements.txt:81`**, `@v0.1.0` to `@v0.2.0`. `git diff -U0` shows that line alone.
- **`CLAUDE.md`**: `:56` now says "pinned at `v0.2.0`" and, after the sentence that names the adopting campaign, "and moved to
  `v0.2.0` on 2026-10-18 in the `rrk-v020-adoption` campaign" (the sentence rewraps over three lines); `:88` says
  "(RunRegistryKit, pinned at `v0.2.0`)". No other line moved.
- **The venv**: `runregistrykit` 0.1.0 at `9f8ebe43…` replaced by 0.2.0 at `0bae3a3c501a8264ee4be5bc24f06422af38720f`, by the
  prompt's two `pip install` commands and no other. The install is not in the commit, and `git revert` does not undo it
  (README §4.2).
- **`measure/out/`** (created): `01-pip.txt`, `01-list.txt`, `01-store-show.txt`, `01-roots.txt`.
- This log, and the board: header, the §1 row for 01, items K1 and B1, the G1 row of §2, `### 5.1 After prompt 01`.

## Deviations from the prompt

1. **`list`'s new clause reads "45 with no host recorded", not 43.** The prompt (§2.4) says *N* "must count every run directory
   that has a `status.json`", which is 43. `list` counts every entry whose status carries no `host` object, and the two
   `a3-pilot` directories have no status at all (`var/runs/a3-pilot/` holds `BACKUP_PATH`, `run.err`, `run.out`, `run.pid`;
   `a3-pilot-resume/` holds `run.err`, `run.out`, `run.pid`), so they count too: 43 + 2 = 45. The note predicted this.
   **STRUCTURALLY REQUIRED** (the count is the package's, not a choice). The count is accounted for, as acceptance 4 requires.
2. **`01-pip.txt` omits the `Requirement already satisfied` lines** of both pip runs (98 each), stated in place, and keeps
   every other line including each `Resolved ... to commit`. **IMPLEMENTATION CHOICE** (the note allows it).
3. **`01-store-show.txt` takes the primary's path from the sidecar's name** (`<stem>.manifest.json` to `<stem>.sqlite`), as
   `runregistry-extraction` 01 did. No `.sqlite` file exists; `store show` reads the sidecar. Each block ends with its exit
   status. **IMPLEMENTATION CHOICE.**
4. **`01-roots.txt` records the load average before each root**, not once. **IMPLEMENTATION CHOICE.**
5. **The frozen baseline blocks are headed by `HEAD` `f0d280a`**, the commit the work started on (the prompt's examples cite
   `8a8cd10`, the commit the orchestrator measured at). Nothing under the code changed between them. **IMPLEMENTATION CHOICE.**
6. **No other deviation.** `README.md`, `prompts/INDEX.md`, `orchestrator/` and anything under `var/` were not touched;
   `docs/OPEN_ISSUES.md` did not need to change.

## Verification performed

- **Baselines before the install** (HEAD `f0d280a`, tree clean, one worktree): `pip freeze`, `package_provenance` (runregistrykit
  0.1.0 at `9f8ebe43…`, `v0.1.0`; datastorekit 0.4.0 at `c8570e84…`; raypoolkit 0.2.0 at `eaef0ce6…`; none editable), `list`
  (exit 0, nothing on stderr, summary `45 run(s) ...: 43 finished, 2 unknown.`), 21 `store show` blocks (all exit 0, all
  `retirement: state: retired`, all `problems: none`), and the seven roots.
- **The dry run**, exit 0: `Would install runregistrykit-0.2.0` and nothing else. It printed `Resolved ... to commit 895e3c70…`
  for RunRegistryKit, the tag object (as the note says), and the tag objects of the two other packages, `8e7c1d77…` and
  `4e7eb153…`, which it did not install.
- **The install**, exit 0: `Uninstalling runregistrykit-0.1.0` (the replacement), `Successfully installed runregistrykit-0.2.0`.
  Nothing else was installed, upgraded or removed.
- **`pip freeze`, before and after:** `diff` is the one line (82), `runregistrykit @ ...@9f8ebe43…` to `...@0bae3a3c501a8264ee4be5bc24f06422af38720f`.
- **`pip check`:** `No broken requirements found.`
- **`package_provenance()` after:** `runregistrykit` 0.2.0, commit `0bae3a3c…`, `requested_revision` `v0.2.0`, `editable` false;
  `datastorekit` and `raypoolkit` identical to before.
- **`pip show`:** version 0.2.0, `Requires:` empty. **`direct_url.json`:** `commit_id` `0bae3a3c…`, `requested_revision` `v0.2.0`, no
  `dir_info`. `importlib.metadata.version` prints `0.2.0`.
- **The core imports nothing heavy:** the check prints `[]`.
- **`TERMINAL_STATES`:** `('done', 'failed', 'killed', 'lost')`. **`this_host()`:** `id` `8d2b247824882486`, `name`
  `mps026088.phys.susx.ac.uk`, `boot` `376461AF-4985-4F42-8E41-F4E914F5E3E6`, `machine` `arm64`, `system` `Darwin`, `python`
  `3.12.15`. All equal the note's; neither `id` nor `boot` is null.
- **`list` after:** exit 0, **nothing on stderr**. The 47 body lines (header and 45 rows, comparing all but the summary) are
  byte-identical to before (`diff`), so each of the 45 runs has the same state and marker. The summary line is `45 run(s) under
  ...: 43 finished, 2 unknown; 45 with no host recorded.` (deviation 1). The `a3-pilot` pair have no `status.json` and no
  manifest; `v0.2.0` reads them `unknown`, with the same "no manifest" note and no `!!`, as the note predicted.
- **`store show` after:** the 21 blocks, compared with the HEAD header removed, are identical to before (21 of 21); all exit 0
  and read `problems: none`.
- **The seven roots**, serially, foreground, README §8's command, load about 5.6 to 8.6:

  | Root | Expected | Before | After | Result |
  |---|---|---|---|---|
  | `AdaptiveLevin/tests` | 32 | 32 | 32 | OK |
  | `CosmologyModels/tests` | 39 | 39 | 39 | OK |
  | `ObjectFactories/tests` | 279 | 279 | 279 | OK |
  | `tools/tests` | 91 | 91 | 91 | OK |
  | `LiouvilleGreen/tests` | 148 | 148 | 148 | OK (1 skipped) |
  | `RunDrivers/tests` | 99 | 99 | 99 | OK |
  | `ComputeTargets/tests` | 700 | 700 | 700 | OK |

  None of README §8's four wall-clock tests failed, so none was re-run.
- **The note's §1 and §2 (prompt §2.6), against what was measured:**
  - `requirements.txt:17` (`datastorekit ...@v0.4.0`), `:75` (`ray==2.43.0`), `:88` (`SQLAlchemy==2.0.39`) hold, and `:81` was
    the `v0.1.0` line it names. The `[stores]` floor `datastorekit>=0.4.0` is met by the installed 0.4.0.
  - "Its existing run directories ... carry no `host`" and "`list` counts them in 'N with no host recorded'": true of the 43 that
    have a status; `list` counts the pair without a status too (deviation 1). The note says it "does not re-count".
  - "05's planner counted 83 run directories in SGK's `var/runs/`": this is quoted as a figure from RunRegistryKit's planning, not a
    claim about now; `var/runs/` holds 45 directories today. Not a false claim, recorded for 03's expectations.
  - "A `stale` one written by a run that was hard-killed ... is ended by `end`": nothing is stale here, so not exercised.

## The deliberate-breakage record

None. The prompt adds no code and no test (README §5 rule 8).

## Observations not acted on

- The note's (§1) 83-run figure and the 45 now are different; if 03 or 04 cite a run count they should use `list`'s.
- `list`'s "no host recorded" clause will keep counting the `a3-pilot` pair and all 43 `v0.1.0` records indefinitely.
  No issue opened.

## State handed to the next prompt

- The venv holds `runregistrykit` 0.2.0 at `0bae3a3c…`; `requirements.txt:81` and `CLAUDE.md:56`, `:88` say `v0.2.0`. The
  rest of `CLAUDE.md`'s registry rules are as they were (02's).
- The registry is 45 runs, 43 finished and 2 unknown, nothing running or stale, no run's reading changed; the 21 stores are all
  tombstones. `this_host()`: `id` `8d2b247824882486`, `boot` `376461AF-4985-4F42-8E41-F4E914F5E3E6`.
- All seven roots pass at §0.2's counts. `docs/OPEN_ISSUES.md` stays at 100.
