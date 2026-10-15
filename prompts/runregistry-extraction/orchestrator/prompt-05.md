# Orchestrator — prompt 05, pin `runregistrykit` at `v0.1.0`

Read [`../README.md`](../README.md) first: §0.1, §2.2 (05's charter), §4.2, §5 (rules 7, 9, 11–14)
and §8. Then the board's §2 (E2), §5.6 (03's review, which this prompt repeats for a new package)
and §5.10. The conventions are [`prompt-01.md`](prompt-01.md)'s §0 last list, carried over, and
[`prompt-03.md`](prompt-03.md) is this note's model; what differs is said here.

**You do not write code.** You may:
- run the suites, `pip` read-only commands, and `python -m RunRegistry list`;
- read RunRegistryKit's repository with `git`, never changing it;
- run a comparison script from the session scratchpad, never committing one;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`05-pin-runregistrykit-at-v0.1.0.md`](../05-pin-runregistrykit-at-v0.1.0.md)
**Board items:** K2 · **Closes:** nothing · **Narrows:** nothing · **Opens:** nothing expected ·
**Model:** Sonnet, as the prompt recommends.

**Gate:**
- E2 holds and 05–08 are written (`17512cb`). 04 landed (`ec61569`) and was reviewed (`e7d18bd`);
  R0's and R1's stores are retired (`aaa7661`). The tree is clean.
- `HEAD` is the commit that adds this note, or a later one that touches no code (outside `prompts/`
  and `docs/`).
- Nothing is `running` in the registry; the rehearsal lock is free; `ray-processes` lists nothing.
- The venv holds no `runregistrykit`.
- One prompt at a time in this checkout (README §4.2).

## 0. What makes this prompt unusual

**It is 03 again, for a package nothing yet imports.** The commit is one added line of
`requirements.txt` and the records. The install happens before the commit, from the working tree,
and `git revert` does not undo it (README §4.2: `pip uninstall -y runregistrykit`). So the review
checks the venv directly, not only the diff.

**Unlike 03, it adds a line rather than moving one,** and nothing SGK runs can see the package:
`RunRegistry/` stays the registry every driver calls until 06. So the roots are the whole of "nothing
SGK runs changed", and there is no fingerprint to take. 03's §2.4 read R0's stores; those stores are
retired (§5.8), and every sidecar under `var/datastores/` is a tombstone.

**What moves on purpose:**
- `requirements.txt` gains one line, after `rpds-py==0.23.1` (`:80`) and before `rsa==4.9`, which
  moves from `:81` to `:82`. No line changes.
- The venv gains `runregistrykit` 0.1.0 at `9f8ebe43…`. Nothing else is installed, upgraded or
  removed.
- `measure/out/05-pip.txt` and `05-roots.txt`.
- `logs/05-pin-runregistrykit-at-v0.1.0.md`.
- The board: header, §1 row 05, K2, `### 5.11 After prompt 05`.
- No issue opens or closes; the index stays at 108.

**A record the prompt contradicts, which the agent does not correct.** README §5 rule 9 says a
manifest's `packages` names `runregistrykit` "(05)". The prompt's §1 is right and the rule is
stale: `RunRegistry.RECORDED_DISTRIBUTIONS` is `("datastorekit", "raypoolkit")`
(`RunRegistry/__init__.py:119`), frozen, so `runregistrykit` enters `packages` at 06. The README is
not on the prompt's file list. The agent records it under "Observations not acted on"; the
orchestrator corrects rule 9 in its follow-up (§4).

**The facts, checked by the orchestrator on 2026-10-15 at `17512cb`** (the tree is `ec61569`'s
outside `prompts/` and `docs/`: `git diff --stat ec61569 17512cb -- . ':!prompts' ':!docs'` is
empty). Pass them on.
- **The release.** Tag `v0.1.0`, object `fd5ce4806a3cea09a27d92d5d811600c62dac376`, commit
  `9f8ebe431503fb8c79ee1407cc460501f1ca2d92`, the same locally and by `git ls-remote origin`.
  Nothing under `runregistrykit/` or in `pyproject.toml` changed from the tag to `origin/main`
  (`caad864`). `pyproject.toml` at the tag: `version = "0.1.0"`, `requires-python = ">=3.12"`,
  `dependencies = []`, `[stores]` = `datastorekit>=0.4.0`, and `runregistrykit.tests` excluded
  from the wheel. The package's modules are `__init__`, `__main__`, `client`, `fingerprint` and
  `stores`.
- **The dry run.** With the line added in a scratch copy of `requirements.txt`, `pip install
  --dry-run -r` printed `Would install runregistrykit-0.1.0` and nothing else. It prints
  `Resolved https://github.com/ds283/RunRegistryKit to commit fd5ce48…`: **that is the tag object,
  not a mismatch**, as 03's log found for DatastoreKit (`8e7c1d77…`). The installed distribution's
  `direct_url.json` records the commit, `9f8ebe43…`, as `datastorekit`'s records `c8570e84…`.
- **The import.** `runregistrykit/__init__.py` imports only the standard library and `.client` at
  module scope; `fingerprint.py` imports `importlib`, `sys`, `pathlib`, `typing` and `.client`, and
  loads DatastoreKit inside `DatastoreKitFingerprinter`'s methods. Importing either reads no config
  file.
- **The venv now.** Python 3.12.15; `datastorekit` at `c8570e849334d09085bbd84876095fc9e22bd005`
  and `raypoolkit` at `eaef0ce60e2fc354bb32f0ad576010e6a83c1976` (`pip freeze`), neither editable;
  no `runregistrykit`. `requirements.txt`: `datastorekit` `:17`, `ray==2.43.0` `:75`, `raypoolkit`
  `:76`, `SQLAlchemy==2.0.39` `:87`. These are the checklist's item 1, which holds.
- **The registry.** 37 runs, 35 finished and 2 `unknown`, none `running` or stale. 18 sidecars
  under `var/datastores/`, all tombstones, and no `.sqlite` file.
- **The roots**, at `17512cb`, serially: all seven pass at 03's counts, with no failure (§1 step 5).
- **Not touched, and staying:** every file under `RunRegistry/`, every test, every `docs/` script,
  the other pins, `README.md`, and everything under `var/`.

**What the review exists to establish.**
- **(P1) The pin.** One line added, to the tag, with the extra; the venv at the tag's commit, not
  editable; nothing else installed, upgraded or removed.
- **(P2) Nothing SGK runs changed.** No SGK file imports the package; importing it loads none of
  `ray`, `sqlalchemy` or `datastorekit`; `package_provenance()` unchanged; the roots at 03's counts.
- **(P3) The checklist's item 1** checked against the venv.
- **(P4) The records** in the campaign's form.

**Conventions,** in addition to `prompt-01.md`'s:
- **The venv.** The agent runs exactly the two `pip install` commands of the prompt (`--dry-run`,
  then the real one) and no other `pip install` or `pip uninstall`. If the dry run shows anything
  but `runregistrykit`, it stops before the real install.
- **No `runregistrykit` command.** No `python -m runregistrykit`, no `store` command through it:
  `runregistrykit.toml` does not exist until 06. The agent's reads are the prompt's §2.3, from the
  repository root.
- **`var/` is not opened.** Nothing in this prompt reads a store or a run directory.
- **Ray.** None. The lock is not taken.
- **The SHA.** "this commit" where 05's hash belongs; the orchestrator substitutes it.

## 1. Before you dispatch

Done at `17512cb` on 2026-10-15; the results are §0's facts.
1. **The tree.** Branch `handover-remedial`; clean; `git worktree list` shows this checkout alone.
2. **The registry.** Nothing `running` or stale.
3. **The lock and Ray.** `tools.rehearsal_lock status`: free; `ray-processes`: none.
4. **The venv and the release.** As §0.
5. **Baselines.** The seven roots, serially, at `17512cb`:

   | Root | Ran | Result | Wall |
   |---|---|---|---|
   | `AdaptiveLevin/tests` | 32 | OK | 1 s |
   | `CosmologyModels/tests` | 39 | OK | 2 s |
   | `ObjectFactories/tests` | 271 | OK | 15 s |
   | `tools/tests` | 91 | OK | 12 s |
   | `LiouvilleGreen/tests` | 148 | OK (skipped=1) | 24 s |
   | `RunRegistry/tests` | 286 | OK | 75 s |
   | `ComputeTargets/tests` | 693 | OK | 139 s |

   Load fell from 33 to 8 over the run (other projects' work); no known flake fired. The agent's
   runs may still show one under load; that is not a stop.

6. **The `var/` marker**, for check 5: immediately before dispatch, `touch` a marker in the
   orchestrator's scratch directory.

## 2. Dispatch

One fresh-context subagent, on **Sonnet**. Give it the prompt, the README and the board, `HEAD`,
the baselines, the index count, and §0 of this note up to "## 1. Before you dispatch", as a copy in
the session scratchpad. The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the end**, staged by explicit path from the prompt's §6 list.
- **The two `pip install` commands, and no other venv change.**
- **The tag object in the dry run** is expected (§0), not a stop.
- **README §5 rule 9** is recorded, not corrected.
- **The wall-clock flakes** are expected, and are re-run alone and recorded, not a stop.
- **Stop and ask** on any of the prompt's §4 conditions; if it stops after the install, say so.

## 3. The review — six checks

1. **Scope.** `git show --stat <05's commit>`: `requirements.txt`, `measure/out/05-pip.txt` and
   `05-roots.txt`, the log and the board; `docs/OPEN_ISSUES.md` only with a new §3 entry.
   `git diff <05>~1 <05> -- requirements.txt` is the one added line, between `rpds-py` and `rsa`.
   Nothing under `RunRegistry/`, `docs/`, `README.md`, `prompts/INDEX.md` or `orchestrator/`.
2. **P1, the venv, measured here.** `pip show runregistrykit` and its `direct_url.json`: 0.1.0,
   `commit_id` `9f8ebe43…`, `requested_revision` `v0.1.0`, no `dir_info`; `pip show -f` lists no
   `runregistrykit/tests/`. `pip freeze` against the agent's 2.1 freeze differs by the one line;
   `datastorekit` and `raypoolkit` unchanged; `pip check` clean; `pip install --dry-run -r
   requirements.txt` now installs nothing. The agent's `05-pip.txt` has the dry run before the
   install.
3. **P2, the imports.** The `sys.modules` check re-run here prints `[]`; `git grep -n runregistrykit
   -- . ':!prompts' ':!docs'` is empty at the agent's commit, `requirements.txt` aside; and
   `package_provenance()` equals the agent's 2.1 reading.
4. **P2, the roots.** `05-roots.txt`: seven roots at 03's counts; any known flake re-run alone and
   both runs recorded. Re-run here at the agent's commit, as §1's table.
5. **Nothing else moved.** `find var -newer <marker>` is empty. `git status --short` clean.
   RunRegistryKit's checkout unchanged (`git -C … status --short`, `HEAD`).
6. **P3 and P4, the records.** Item 1's five versions checked against the venv, any false one
   recorded. The board's header, §1 row, K2 and §5.11; the log's §5.1 headings and classified
   deviations, with rule 9 under "Observations not acted on"; the commit message in `CLAUDE.md`'s
   form, ending `Co-Authored-By: Claude Sonnet …`.

## 4. After the review

- **Record it on the board** as an *Orchestrator review of prompt 05* paragraph in §5.11, with
  residue fixed in a follow-up commit: "this commit" → the SHA; this note listed as "used for 05";
  README §5 rule 9's "(05)" → "(06)"; `prompts/INDEX.md`'s row (6 of 10 landed).
- **Report to the user:** the pin and the venv; the import check; the roots; what 06 inherits.
- **Then 06**, the swap. Its note is written after this review.
