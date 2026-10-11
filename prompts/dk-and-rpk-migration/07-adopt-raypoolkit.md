# Prompt 07 — adopt `raypoolkit`

**Campaign:** [`README.md`](README.md) · **Board items:** **B1**–**B3** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-11, with prompts 05, 06, 08 and 09, after the review of 04 (`0b6565e`).
**Gate:**
- 06 landed and reviewed; `raypoolkit` 0.2.0 is installed from the pin at `eaef0ce…`, not
  editable (board, 06's row);
- `git status` is clean;
- nothing is `running` in the registry.

**Closes:** nothing by name. **Opens:** anything you find.
**Recommended model:** **Sonnet**. Two files deleted and seven identical import lines rewritten,
with the proof that every pool still constructs and nothing else moved.

**Read first:**

1. [`README.md`](README.md): §1, §2.2 (07's charter), §4.3, §5, and §6.2 (C7, C8, C10).
2. **RayPoolKit's SGK checklist**, `/Users/ds283/Documents/Code/RayPoolKit/docs/adoption/secondarygwkit.md`,
   items 2–6, 8 and 9. Its `path:line`s are SGK's at `b510bc9`. **Read it; do not edit anything in
   that repository.**
3. RayPoolKit's `docs/adoption/README.md` §2 (the module map) and §5 (the order with DatastoreKit),
   and its `README.md`, "Which handlers a pool needs" and "Changes in `v0.2.0`".
4. [`logs/04-adopt-datastorekit.md`](logs/04-adopt-datastorekit.md): how 04 recorded a swap, its
   §3.1-style measurement at a scratch commit, and its breakage diffs. 07 is the same shape, much
   smaller.
5. `CLAUDE.md`, "The datastore layer is `datastorekit`", whose last paragraph this prompt replaces.
6. `RunRegistry/tests/test_pipeline_adoption.py:31-37`, the comment this prompt corrects.

---

## 1. What is wanted

SGK's work queue, `RayTools/RayWorkPool.py`, is replaced by the installed package. `v0.1.0` of the
package was that file under a new name and `v0.2.0` is `v0.1.0` with its inherited faults fixed
(RayPoolKit `PROVENANCE.md`). **No call site changes** (C7), and the name every call uses,
`RayWorkPool`, stays bare. The swap is one commit.

- **B1 — `RayTools/` goes.** `git rm RayTools/RayWorkPool.py RayTools/__init__.py`. With them goes
  the one line 04 rewrote under U12 (`:8`). `utilities.format_time` stays SGK's.
- **B2 — the seven imports.** Each `from RayTools.RayWorkPool import RayWorkPool` becomes
  `from raypoolkit import RayWorkPool`, and nothing else on the line or in the file changes:

  | File:line at `0b6565e` |
  |---|
  | `main.py:63` |
  | `extract_GkSource_data.py:37` |
  | `extract_GkWKB_data.py:35` |
  | `extract_Gk_data.py:32` |
  | `extract_QuadSourceIntegral_data.py:42` |
  | `extract_TkWKB_data.py:30` |
  | `extract_tensor_source_data.py:31` |

  Re-measure at your `HEAD` first (§3.1); its figures govern. Seven lines need no script: edit
  them, and show by `git diff -U0` that each hunk is that one line. None is nested, none names a
  private name, and no `import_module` call names `RayTools` (checklist item 3).
- **B3 — the prose this commit makes false** (C10):
  1. **`CLAUDE.md`.** Its section "The datastore layer is `datastorekit`" ends with a paragraph
     saying `RayTools/RayWorkPool.py` is frozen while it moves to RayPoolKit. Complete the section
     for both packages, as README §4.3 says:
     - retitle it **"The datastore layer is `datastorekit`; the work queue is `raypoolkit`"**;
     - its first paragraph names `raypoolkit` too: RayPoolKit (`/Users/ds283/Documents/Code/RayPoolKit`,
       https://github.com/ds283/RayPoolKit), pinned at `v0.2.0`, adopted in this campaign;
     - "It is not this repository's code" covers both packages, each reported to its own
       repository;
     - "Imports name the package" adds `from raypoolkit import RayWorkPool`, and says that the
       name stays bare, because five `ComputeTargets/tests` modules find pool constructions by
       `ast.Name`;
     - the last paragraph, on `RayTools/`, goes.

     Keep it no longer than it is now, plus three lines. Nothing else in `CLAUDE.md` changes; its
     test command is 09's.
  2. **`RunRegistry/tests/test_pipeline_adoption.py:31-33`.** The comment gives the printed lines'
     places as `:166`, `:548` and `:566`, line numbers of the deleted copy. Make them the
     package's at `v0.2.0`: the banner `raypoolkit/RayWorkPool.py:206`, a progress `print(msg)`
     `:588`, the completion `print(` `:606` (checklist item 6; check each against the installed
     file). The same sentence says the block holds "six lines" including "the completion pair";
     the block holds five, the banner and two progress notifications of two lines each, and no
     completion line. Since the sentence is being rewritten, make it say what the block holds.
     Change nothing else in the file: `A3_OUTPUT` and every test stay byte for byte.
  3. **Nothing else.** `docs/tolerance-convergence/inventory.py:382` keeps `"RayTools"` in
     `PRODUCTION_TREES` (C10). Dated `docs/` records and `prompts/` are left. If you find other
     prose outside them that names `RayTools/` as a current path, record it; do not fix it.

**What does not change.** Every pool site and every handler keyword, the 44 lookup-only sites'
`store_handler=None, persist_handler=None` included (C7). `main.py` but `:63`. Every test but the
one comment. The datastore layer and its pin. What the code writes: proved by R2 (08).

---

## 2. Behaviour that changes, and where it is measured

RayPoolKit's checklist item 5 lists what `v0.2.0` changes. One site meets a change:
**`main.py:3247`**, the `APPLY GKSOURCE POLICIES` queue, whose `post_handler` now gets the object
the persist returned, not the local copy. By reading, the statistics it records are the same. 07
does not measure it: 08 does, from the rehearsal's statistics block (C8). Every other row of the
checklist's table is "nothing" for SGK; check each against the tree and say so in the log.

The printed lines are an interface: `RunRegistry`'s `StageTracker` parses the banner, progress and
completion lines, and `test_pipeline_adoption.py` pins five of them. The package prints them as
SGK's copy did (checklist item 5, "The printed lines"). `test_pipeline_adoption` passing is the
test-level evidence; 08's registry fields are the run-level evidence.

---

## 3. Verification

### 3.1 By RayPoolKit's script

From RayPoolKit's root, with its venv:
`./venv/bin/python docs/adoption/measure_client_pools.py sgk --commit <SHA>`.
1. **At your `HEAD`, before any change.** Save it under `var/dk-and-rpk-migration-07/`. Expected
   (the orchestrator measured `f620f58` on 2026-10-11): 7 imports, all at module level; 54
   `RayWorkPool(...)` calls, verdict `constructs` 54; 2 string literals naming `RayTools`
   (`docs/tolerance-convergence/inventory.py:382`, and a `prompts/` record); 0 `import_module`
   calls; 6 test modules naming `RayTools` or `RayWorkPool` (04 deleted the seventh,
   `test_layer_is_generic.py`). `RayTools/RayWorkPool.py` is blob `3c1d9bd…`, `b4666fa…` with
   04's U12 line.
2. **At a commit of the staged tree.** As 04 did: `git add` the work, then
   `c=$(git commit-tree $(git write-tree) -p HEAD -m "07 scratch")`, and measure `--commit $c`.
   Never commit to or amend the branch before §5. Save the Markdown as
   `measure/out/07-raypoolkit-measure_client_pools.md`. It must show:
   - `RayTools/` absent at the commit;
   - **0 imports** and **0 `import_module` calls** naming `RayTools`;
   - string literals: the 2 above and no others;
   - **54 calls, every one `constructs`**, with the same handler-keyword table as at `HEAD`.

   `git diff --stat <scratch> <commit>` then lists only the records.
3. `git ls-tree <commit> RayTools` is empty.

### 3.2 The suites

All six roots, serially, in the foreground; Datastore in its four groups. Every count is 04's:
Datastore 320 (54 + 103 + 101 + 62), RunRegistry 286, ComputeTargets 693, AdaptiveLevin 32,
CosmologyModels 39, LiouvilleGreen 148 (1 skipped).
- Name the five `ComputeTargets/tests` modules of checklist item 6 and show each passes:
  `test_main_plumbing`, `test_quadsource_policy_main`, `test_wkb_numeric_parent_main`,
  `test_gksource_parent_set_main`, `test_quadsource_integral_parent_main`. And
  `RunRegistry/tests/test_pipeline_adoption`.
- `black --check` is clean on every Python file touched.

### 3.3 Through the package, without Ray

1. `py_compile` on `main.py` and the six `extract_*.py`. Never import them: they call `ray.init` at
   module scope.
2. In a fresh `./venv/bin/python -I` with the repository on its path: `from raypoolkit import
   RayWorkPool`; `RayWorkPool.__module__` is `raypoolkit.RayWorkPool`; `import RayTools` raises
   `ModuleNotFoundError`; `ray.is_initialized()` is false.
3. `PYTHONPATH=. ./venv/bin/python -m RunRegistry list` runs.

### 3.4 Deliberate breakage (README §5 rule 8)

Each must make the named check fail; then restore it.
- (i) **One call spelled `raypoolkit.RayWorkPool(...)`**, with `import raypoolkit` added beside
  the import, at a site one of the five `ast`-matching modules inspects (for example a
  `RayWorkPool(...)` inside a function `test_quadsource_policy_main` or `test_main_plumbing` reads).
  Name the site, and show which test fails and how. If no test fails, that is a finding: record it,
  and try a site another module inspects.
- (ii) **One `extract_*.py` import reverted** to `from RayTools.RayWorkPool import RayWorkPool`:
  the script, at a scratch commit made from a **temporary index** (`GIT_INDEX_FILE` in your
  scratch directory; never the checkout's index), counts 1 import. No test imports an
  `extract_*.py`, so the script is the check; say so.
- (iii) **`main.py:63` reverted**: the script counts 1 import, and `test_main_plumbing` still
  passes, since it reads `main.py` by `ast` and never runs its imports. Record both: it shows why
  the script, not the suite, is B2's proof.
- (iv) **`test_pipeline_adoption`'s `A3_OUTPUT` banner changed** (`** CALCULATE QUADRATIC SOURCE
  INTEGRALS` → `* CALCULATE …`): the test that feeds it to `StageTracker` fails. This shows the
  pinned lines still bite after the swap.

Record each in the log as a diff, exactly as applied, with what failed. Check each with
`git apply --check` and `-R --check`, replay in `bash`, and leave `git status` clean.

---

## 4. Acceptance

1. **B1–B3** as §1 says.
2. §3.1's measurement at a commit of the staged tree: `RayTools/` absent, 0 imports, 0
   `import_module` calls, the 2 strings, 54 calls all `constructs`.
3. §3.2's counts, the six named modules passing, and `black --check` clean.
4. §3.3's checks, with Ray never started.
5. §3.4's four breakages, each caught as stated.
6. **The records**, in the same commit:
   - this board: the §1 row for 07, items B1–B3, and `### 5.7 After prompt 07 (<date>)`, naming
     the rollback: `git revert` alone, since an unimported package is inert (README §4.2);
   - README §4: a line "07 ships": `RayTools/` gone, the seven imports, `CLAUDE.md`'s section
     completed, as named or naming what differs;
   - a §3 entry for anything opened, with `docs/OPEN_ISSUES.md` (§1.20) in the same commit.

---

## 5. Stop conditions — stop and ask

- **Any call's verdict is not `constructs`**, or the script finds a number of calls other than 54
  at your `HEAD` or at the scratch commit.
- An import of `RayTools` that is not one of the seven, or nested, or through `import_module`.
- A staying test fails for any reason but an import B2 should have rewritten. Never edit a test's
  assertions to pass; the comment of B3.2 is the only test change.
- The swap would need a change at a call site, a handler, `main.py` beyond `:63`, or anything in
  RayPoolKit or DatastoreKit.
- Anything would start Ray, or open a store.

---

## 6. What this prompt does not do

- **Files it changes:** the two deletions (B1); the seven importing files (B2); `CLAUDE.md` and
  `RunRegistry/tests/test_pipeline_adoption.py` (B3); new
  `measure/out/07-raypoolkit-measure_client_pools.md`; the log, this board, README §4's line and,
  if an issue is opened, `docs/OPEN_ISSUES.md`. Nothing else.
- It changes no pool site, no handler keyword, and no `docs/` file.
- It runs no build. R2 is 08.

---

## 7. The log and the board

`logs/07-adopt-raypoolkit.md`, using README §5.1. In addition, it gives:
- §3.1's two measurements, against the checklist's appendix and the figures above;
- the seven `-U0` hunks;
- checklist item 5's table, row by row, against the tree;
- §3.2's table;
- each breakage (i)–(iv), as a diff, with what failed;
- **for 08:** the commit, the venv's `package_provenance()`, and what R2's launches should expect
  to differ from R1's: the post handler at `main.py:3247`, and anything in the printed lines.

**One commit**, in `CLAUDE.md`'s form, staged by explicit path only, after showing
`git diff --cached --name-only`. Deletions are staged with `git rm`.
