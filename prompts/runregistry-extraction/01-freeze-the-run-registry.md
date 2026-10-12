# Prompt 01 — freeze the run registry while it moves to RunRegistryKit

**Campaign:** [`README.md`](README.md) · **Board items:** **F1**–**F3** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-12, with the plan and prompt 02, at `c8a7f14`.
**Gate:**
- the plan's commit is `HEAD` or an ancestor of it, and `git status` is clean;
- `python -m RunRegistry list` shows nothing `running`.

**Closes:** nothing. **Opens:** only what the work finds.
**Recommended model:** **Sonnet**. A short section of `CLAUDE.md`, a table of blob hashes, and three
saved measurements.

**Read first:**

1. `CLAUDE.md` in full, and especially "The datastore layer is `datastorekit`; the work queue is
   `raypoolkit`" and "Long-running jobs — the run registry".
2. [`README.md`](README.md): §0, §0.2, §2.2 (01's charter), §4 (01's line), §5 (rules 9, 10 and 14)
   and §6.2 (C4).
3. [`docs/runregistry-extraction-audit.md`](../../docs/runregistry-extraction-audit.md) §1 and §2.
4. The board's §2.1, which this prompt fills.
5. [`runregistrykit-extraction-brief.md`](runregistrykit-extraction-brief.md) §2, for what the frozen
   tree is for.

---

## 1. What is wanted

RunRegistryKit will import `RunRegistry/` from this repository (README §0; the brief, §2). The
import is checked file by file against the tree as it stood when it was frozen, its G1. So from this
commit until prompt 06 deletes it, **`RunRegistry/` does not change**, and the freeze has to be
written where every later session reads it: `CLAUDE.md`. The previous freeze of this kind was the
datastore layer's, at `b510bc9`, which `dk-and-rpk-migration` 04 later replaced.

This prompt also takes the three baselines that 06 and 08 compare with, while the registry that
wrote every record is still the one reading it.

## 2. What to do

**2.1 `CLAUDE.md`.** Add a section directly after "Long-running jobs — the run registry", titled
**"`RunRegistry/` is frozen — it is moving to RunRegistryKit"**. In a few lines, say:
- since this commit (name it as "prompt 01 of `runregistry-extraction`"; the hash is not known
  until the commit exists) `RunRegistry/` is frozen, because RunRegistryKit imports it and checks
  the import against the frozen tree;
- nothing under `RunRegistry/`, its tests included, is changed until
  `prompts/runregistry-extraction` prompt 06 replaces it with the pinned `runregistrykit`;
- a defect found in it is recorded on that campaign's board and in `docs/OPEN_ISSUES.md`, and is
  fixed in RunRegistryKit, not here; if a fix cannot wait, stop and ask the user;
- the six rules of the section above still bind, unchanged.

Change nothing else in `CLAUDE.md`.

**2.2 The frozen tree.** Fill the board's §2.1 with every tracked file under `RunRegistry/` at
`HEAD` (which, `RunRegistry/` being unchanged by this commit, is the frozen tree), from
`git ls-tree -r HEAD RunRegistry/`: path and full blob hash, in path order. Count the rows and
compare with `git ls-tree -r --name-only HEAD RunRegistry/ | wc -l` (25 at `c8a7f14`). `__pycache__`
is untracked and is not listed.

**2.3 The baselines**, each saved under `prompts/runregistry-extraction/measure/out/`:
- **`01-list.txt`**: `PYTHONPATH=. ./venv/bin/python -m RunRegistry list`, verbatim.
- **`01-store-show.txt`**: `python -m RunRegistry store show <primary>` for each of the 14 sidecars
  under `var/datastores/` (`find var/datastores -name '*.manifest.json'`), the primary path being
  the sidecar's name with `.manifest.json` replaced by `.sqlite`; the A3 backup's is under
  `backup-pre-resume-20260921T091011/`. One block per store,
  headed by the command. Read only: nothing under `var/` is written.
- **`01-roots.txt`**: each of the seven roots run once, in the foreground, serially, by README §8's
  command, with the `Ran`, `OK`/`FAILED` and any `FAIL:`/`ERROR:` lines of each.

Record the `HEAD` each was taken at.

## 3. Acceptance

1. `git diff --stat c8a7f14 HEAD -- RunRegistry/` is empty before you commit, and the staged
   change touches nothing under `RunRegistry/`.
2. §2.1's table has one row per tracked file, its count equal to `git ls-tree`'s, every blob hash
   copied from `git ls-tree`.
3. `01-list.txt` shows README §0.2's 29 runs, 27 finished and 2 `unknown`, none `running`.
4. `01-store-show.txt` reads each of the 14 sidecars as a tombstone, with no problem reported. A
   sidecar read otherwise is recorded and opened as a §3 issue, not repaired.
5. `01-roots.txt`: the seven roots pass at README §0.2's counts. A failure among README §8's four
   known wall-clock tests is re-run alone and recorded; any other failure stops the prompt (§4).
6. **No deliberate breakage**: this prompt adds no code and no test.

## 4. Stop conditions — stop and ask

- Anything under `RunRegistry/` differs from `c8a7f14`'s.
- `list` shows a run `running` or stale.
- A root fails other than by README §8's known tests, or a count differs from §0.2's.

## 5. What this prompt does not do

- It changes nothing under `RunRegistry/`, `var/` or any other repository.
- It does not write the briefs' consequences into `CLAUDE.md` beyond §2.1's section.
- It does not run a build (02 does).

## 6. The log and the board

**The files you may change, and nothing else:** `CLAUDE.md`; the three files under
`measure/out/`; the log; this campaign's board; and `docs/OPEN_ISSUES.md` only if §3 gains an entry.

**The log**, at `logs/01-freeze-the-run-registry.md`, in README §5.1's form.

**The board:** the §1 row for 01, items F1–F3, §2.1's table, and `### 5.1 After prompt 01 (<date>)`.

**One commit**, in `CLAUDE.md`'s form, staged by explicit path only, after showing
`git diff --cached --name-only`.
