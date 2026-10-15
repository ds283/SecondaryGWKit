# Log — runregistry-extraction prompt 05, pin `runregistrykit` at `v0.1.0`

- **Subject:** [`05-pin-runregistrykit-at-v0.1.0.md`](../05-pin-runregistrykit-at-v0.1.0.md), board item K2.
- **Commit:** `3dcab1b` (parent `e381906`).
- **Result:** landed. K2 passes; no stop condition of the prompt's §4 fired. All seven roots pass at 03's
  counts with no failure, so no known flake needed a re-run.
- **Model:** Sonnet 5.5.
- **Date:** 2026-10-15.

The orchestrator's facts for this prompt (its note's §0) were read first and are cited as "the note".

## What shipped

- **`requirements.txt:81`**, one added line, after `rpds-py==0.23.1` and before `rsa==4.9` (now `:82`):
  `runregistrykit[stores] @ git+https://github.com/ds283/RunRegistryKit@v0.1.0`. `git diff -U0` shows that line alone.
- **The venv**: `runregistrykit` 0.1.0 at `9f8ebe431503fb8c79ee1407cc460501f1ca2d92`, installed from the working tree by
  the prompt's two `pip install` commands and no other. The install is not in the commit and `git revert` does not undo
  it (README §4.2: `pip uninstall -y runregistrykit`).
- **[`measure/out/05-pip.txt`](../measure/out/05-pip.txt)** and **[`05-roots.txt`](../measure/out/05-roots.txt)**.
- This log, and the board: header, the §1 row for 05, item K2, `### 5.11 After prompt 05`.

## Deviations from the prompt

1. **`05-pip.txt` omits the `Requirement already satisfied` lines** of both pip runs (98 lines each), headed so, as
   `03-pip.txt` does. **IMPLEMENTATION CHOICE.** Every other line is kept, including the three `Resolved ... to commit` lines.
2. **§2.3's `git grep -n 'runregistrykit'` outside `prompts/` and `docs/` is not empty.** It gives one line before and
   after the install, `CLAUDE.md:114`, prose in the freeze section saying that prompt 06 replaces `RunRegistry/` "with the
   pinned `runregistrykit`"; after the edit it also gives `requirements.txt:81`, the pin itself. The prompt's claim was meant
   for code. With `':!CLAUDE.md'` the search is empty before, and gives only the pin after; no `.py` file names the package.
   Both forms are in `05-pip.txt`. **STRUCTURALLY REQUIRED** (the literal pattern cannot return an empty result on this tree,
   and the pin must name the package; the intent, that nothing imports it, is met).
3. **`05-roots.txt`'s header records the load only after the last root** (average 27.9), not through the run.
   **IMPLEMENTATION CHOICE.**
4. **No other deviation.** Nothing outside §6's list was changed. `README.md`, `prompts/INDEX.md`, `orchestrator/`,
   `RunRegistry/` and `var/` were not edited, and the orchestrator's directory was not read.

## Verification performed

- **The dry run (§2.3).** `pip install --dry-run -r requirements.txt`, exit 0: `Would install runregistrykit-0.1.0` and
  nothing else. It printed `Resolved https://github.com/ds283/RunRegistryKit to commit fd5ce4806a3c…`, the tag object, as the
  note and 03's log say; the installed record gives the commit.
- **The install.** `Successfully installed runregistrykit-0.1.0`, no uninstall, no other package.
- **Provenance.** `pip show runregistrykit`: version 0.1.0, `Requires:` empty (no core dependency), `Required-by:` empty.
  `direct_url.json`: `{"url": "https://github.com/ds283/RunRegistryKit", "vcs_info": {"commit_id":
  "9f8ebe431503fb8c79ee1407cc460501f1ca2d92", "requested_revision": "v0.1.0", "vcs": "git"}}` — no `dir_info`, so not editable.
  `importlib.metadata.version('runregistrykit')` prints `0.1.0`.
- **`pip freeze`, before and after.** `diff` is the one added line (82):
  `runregistrykit @ git+https://github.com/ds283/RunRegistryKit@9f8ebe431503fb8c79ee1407cc460501f1ca2d92`. `datastorekit` stays at
  `c8570e84…`, `raypoolkit` at `eaef0ce6…`.
- **`pip check`:** `No broken requirements found.`, exit 0.
- **`package_provenance()` before and after:** byte-identical (`cmp`): `datastorekit` 0.4.0 at `c8570e84…`, `v0.4.0`, not
  editable; `raypoolkit` 0.2.0 at `eaef0ce6…`, `v0.2.0`. It does not name `runregistrykit`, as expected (below).
- **The core imports nothing heavy:** `import sys, runregistrykit, runregistrykit.fingerprint; print(sorted(n for n in ('ray',
  'sqlalchemy', 'datastorekit') if n in sys.modules))` prints `[]`.
- **The seven roots (§2.4)**, serially, foreground, load average 27.9 after the last, against 03's counts:

  | Root | Expected | Ran here | Result |
  |---|---|---|---|
  | `AdaptiveLevin/tests` | 32 | 32 | OK |
  | `ComputeTargets/tests` | 693 | 693 | OK (no flake this time) |
  | `CosmologyModels/tests` | 39 | 39 | OK |
  | `LiouvilleGreen/tests` | 148 | 148 | OK (skipped=1) |
  | `ObjectFactories/tests` | 271 | 271 | OK |
  | `RunRegistry/tests` | 286 | 286 | OK |
  | `tools/tests` | 91 | 91 | OK |

- **The checklist's item 1 (§2.5)**, against the venv and `requirements.txt`: Python 3.12.15 (`venv/pyvenv.cfg`, and the
  interpreter that ran every command here); `ray==2.43.0` (`requirements.txt:75`, `pip freeze`); `SQLAlchemy==2.0.39` (`:87`);
  `datastorekit` 0.4.0 at `c8570e84…`; `raypoolkit` 0.2.0 at `eaef0ce6…`. **All five hold.** The pin line the checklist gives is
  exactly the line added. Items 2 onward are 06's, 07's and 08's.
- **`black`:** nothing here is Python; not run.
- **Scope:** `git diff --cached --name-only` is shown before the commit (below). Nothing under `RunRegistry/` or `var/` is changed.

## The deliberate-breakage record

None. This prompt adds no code and no test (README §5 rule 8; acceptance 5).

## Observations not acted on

- **README §5 rule 9 is stale.** It says a manifest's `packages` names `runregistrykit` "(05)". The prompt's §1 is right:
  `RunRegistry.RECORDED_DISTRIBUTIONS` is `("datastorekit", "raypoolkit")` and frozen, `package_provenance()` is unchanged by this
  install, and `runregistrykit` enters `packages` at 06. The README is not on this prompt's file list; the orchestrator corrects it.
- **The prompt's `git grep -n 'runregistrykit'` matches `CLAUDE.md:114`** (deviation 2). 06 and any later check that states
  "nothing imports it" want a pattern that excludes prose, e.g. `':!CLAUDE.md'`, or `'*.py'`.
- **`pip install --dry-run` prints `Resolved ... to commit fd5ce480…`, the tag object, not `9f8ebe43…`.** The installed
  `direct_url.json` and `pip freeze` record `9f8ebe43…`. pip's wording, as 03 noted for DatastoreKit.
- **`pip show runregistrykit` has no `Requires:`** even though `[stores]` was requested: the extra's only dependency,
  `datastorekit>=0.4.0`, is already installed and pip does not list extras' dependencies there. `Required-by:` is empty.
- Nothing opened, so `docs/OPEN_ISSUES.md` stays at 108.

## State handed to the next prompt

- **`requirements.txt:81`** pins `runregistrykit[stores]` at `v0.1.0`, and the venv holds it: `9f8ebe43…`, not editable.
  `datastorekit` and `raypoolkit` are as before. `pip freeze` differs from before by that one line.
- **Nothing imports it.** `RunRegistry/` stays the registry every driver calls until 06; a manifest written before 06 records
  only `datastorekit` and `raypoolkit`. There is no `runregistrykit.toml`, so no `runregistrykit` command works yet (06 writes it).
- **The roots pass at 03's counts.** The lock is free, the registry shows nothing `running` (37 runs, 35 finished, 2 `unknown`),
  and this prompt took no lock, started no Ray and wrote nothing under `var/`.
- **Rollback** (README §4.2): `git revert` of `3dcab1b`, then `./venv/bin/pip uninstall -y runregistrykit`.
