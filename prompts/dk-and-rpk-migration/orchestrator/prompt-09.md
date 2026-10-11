# Orchestrator — prompt 09, close the migration

Read [`../README.md`](../README.md) first: §2 and §2.2 (09's charter), §4.3, §5, §6.2 (C2, C9–C12)
and §6.3. Then this board's §2, §3 and the *Orchestrator review of prompt 08*, and
[`prompt-04.md`](prompt-04.md), whose review of a sweep of imports this note's review of a sweep of
prose follows.

**You do not write code.** You may:
- run both packages' measuring scripts from their roots, with their own venvs, at any SGK commit;
- run the suites, the §2.1 regex and the import check of §0 from the session scratchpad;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`09-close-the-migration.md`](../09-close-the-migration.md)
**Board items:** Z1–Z7 · **Closes:** `[04-comments-name-the-layers-old-module-paths]`,
`[04-six-docs-probes-import-factories-from-the-layer]` (Z7, chosen), and on their own boards, as
*transferred to DatastoreKit*, `[00-a-drop-action-drops-a-replicated-table-outside-the-in-flight-record]`,
`[00-a-new-stores-first-open-can-leave-a-primary-without-its-shards]` and
`[00-the-inventory-cannot-see-a-serial-split]` · **Narrows:**
`[01-cross-filesystem-move-advice-says-delete-by-hand]` (by the user's decision, below) ·
**Opens:** nothing expected · **Model:** Sonnet, as the prompt recommends.

**Gate:**
- 08 landed (`f539015`) and was reviewed (`ede4062`). Both G2s hold on this board's §2. The tree is
  clean.
- `HEAD` is the commit that adds this note, or a later one that touches no code.
  - "Code" here means anything outside `prompts/`.
  - If a later commit touches code, re-run §2.1's regex and both measurements at the new `HEAD`, and
    hand the agent the corrections as an addendum.
- Nothing is `running` in the registry, and no Ray process of this repository is up. 09 runs no
  build and starts no Ray, but the suites must not share the machine with one.
- **The user's decisions of README §6.3 are taken** (§0, "The user's decisions").

## 0. What makes this prompt unusual

**It is records and prose, and one mechanical sweep.** No line of code changes outside the six
probes' import statements (Z7), and no test changes. So the review has three halves:
- **the prose sweep is exact**: 23 lines, each changing a path and nothing else, every touched
  module `ast`-equal with docstrings removed, and §2.1's regex empty after it and 23 lines before;
- **the hand-over is right**: three entries moved whole to their boards' §4, one narrowed by the
  user's decision, each board touched only there, the index and every count agreeing by counting;
- **the close is true**: the final measurements at the agent's commit, the G2 drafts mapping to each
  package's own checklist, the six roots at 08's counts.

**The user's decisions (2026-10-12, at this note's writing; README §6.3 and C11).** Record both in
the log and on the board, as the user's.
- **Z7: repair the six probes.** Chosen. The fix is the prompt's, one statement each. So
  `[04-six-docs-probes-import-factories-from-the-layer]` closes.
- **`[01-cross-filesystem-move-advice-says-delete-by-hand]`: narrow it**, not transfer it whole. The
  orchestrator put the reading to the user, and the user chose to narrow:
  - **The text is the package's.** `ShardedPool._failure_message` appends, on `EXDEV`, "copy it
    instead, and then delete the source by hand" (installed `datastorekit/SQL/ShardedPool.py:3026-3030`;
    `move_store`'s docstring says the same at `:2683-2684`). The package has
    `ShardedPool.delete_store` (`:2704`), so the advice to delete by hand is the package's own
    defect. This half goes to DatastoreKit, whose `docs/OPEN_ISSUES.md` §1.2 already lists the issue.
  - **SGK relays it about a registered store.** `RunRegistry.stores.move_store`
    (`RunRegistry/stores.py:1109-1123`) calls `_store_files("move", ShardedPool.move_store, …)` at
    `:1123`; `_store_files` (`:1052-1060`) re-raises `f"{e}. No sidecar was written or moved: …"`,
    the package's message verbatim with the sidecar listing appended. SGK adds no advice and shapes
    none. **But the harm is SGK's rule**: a registered store's source is retired with `store retire`,
    never deleted by hand (store-retirement decision 6.1.1), and the package cannot know a store is
    registered. That half stays: `store move` passes on, for a registered store, advice its rule
    forbids, and the place SGK would say otherwise is `_store_files` or `move_store`.
  - **The board entry's "Next step"** names `Datastore/tests/test_copy_move_store.py`, which asserts
    the text verbatim. 04 deleted it from SGK; it is now the package's
    (`datastorekit/tests/test_copy_move_store.py`, DatastoreKit's `docs/extraction-verification.md:1165`).
    So the test constraint goes with the package's half, and SGK's half has none.
  - So the entry stays in store-retirement's §3, gains a `**Narrowed (2026-10-12, dk-and-rpk-migration
    prompt 09):**` line saying both halves as above, and its index row stays in §1.14 with its hook
    rewritten to SGK's half. The agent still shows the reading with these lines (prompt §2.5), at its
    `HEAD`.
- **The other three of §6.3 stay the user's, after 09's review:** retiring the six rehearsal stores
  (README §3), the two G2 commits in the package repositories (C12), and whether to rename the
  `Datastore` package (U3). §5.9 lists them as left for the user. None is an issue.

**Corrections to the prompt.** The orchestrator checked the prompt against the tree at `ede4062` and
against both installed packages. Each correction below governs, and the agent logs it as an
`IMPLEMENTATION CHOICE` citing this note, or `STRUCTURALLY REQUIRED` where the prompt cannot be
followed as written.

1. **Z7's check cannot be 04's method** (prompt §1, Z7, "five with no arguments"). 04 ran five
   probes with no arguments because each failed at its import. Once repaired, a run goes on into the
   probe's body: `inventory.py` with no arguments runs its `main()` (`:1419`, `:1454`), and the
   others build schemas and print. **Run no probe.** Show that each gets past its imports by
   executing its import statements and nothing else, with the orchestrator's script (§0, "The import
   check"), from a copy in the agent's scratch directory:
   - before the repair, on the six: each fails on its `_factories` line, which is the non-vacuity;
   - after it, on the six: each `all OK`;
   - after it, on the six that fail through `lookup_keys/common.py` (`probe_01`, `01b`, `04`, `04b`,
     `05`, `06`): each `all OK`. `probe_01b` and `04b` import `probe_01` and `probe_04` as modules;
     both guard their bodies with `if __name__ == "__main__"` (`:249`, `:172`), so nothing runs.

   The script is not committed (acceptance 5: 09 adds no code); the log quotes it verbatim.
2. **The user's decision on the move advice** replaces the prompt's "Read the path … either way"
   (Z1): narrow, as above. The agent records the reading, not a choice.
3. **`config/datastore.py:4`** (Z2's last bullet). The line names the layer as ``Datastore/SQL/``,
   ``Datastore/store_reader.py`` and ``Datastore/store_inventory.py``, in one parenthesis. Rewrite
   all three in file-path form: ``datastorekit/SQL/``, ``datastorekit/store_reader.py``,
   ``datastorekit/store_inventory.py``, not the prompt's dotted `datastorekit.SQL` beside two file
   paths. `Datastore/SQL/` still exists in SGK (the factories), so the old text misnames the layer
   rather than naming a deleted path; §2.1's regex reaches line 4 through `Datastore/store_reader.py`.
4. **Z2 is 04's 23 lines, and only those.** At `ede4062` they are at the board's line numbers
   exactly, in 18 files. 07 recorded no `RayTools` prose outside `docs/` and `prompts/` (07's log,
   "Observations not acted on"), and the regex finds none.
5. **README §2's table** (Z6, "marks 05–09 landed"). Every row, 01–08 included, still says "written
   2026-10-11"; no prompt marked itself. Mark all nine landed, each with its commit, in the board's
   §1 form. The prompt assumed 01–04 were marked.
6. **DatastoreKit's measurement** (Z4, "0 imports"). The script counts `prompts/` too. At `ede4062`
   it gives **51** layer import statements in 31 files, **20** string literals and **1**
   `import_module` call, **all in `prompts/`**, and none outside. "0 imports" is "0 outside
   `prompts/`"; the saved output and §4 say both numbers.
7. **DatastoreKit has released `v0.2.3`** (2026-10-11, `0de2b16`, its `factory-defaults` campaign).
   SGK stays pinned at `v0.2.2`, which DatastoreKit's G2 admits ("`v0.2.2` or later"), and 09 moves no
   pin. DatastoreKit's checkout is at `83068cb`; its `docs/extraction/measure_client_imports.py` is
   unchanged since `v0.2.2` (`git diff v0.2.2 HEAD -- docs/extraction/measure_client_imports.py` is
   empty), so running it from that checkout is running the tag's. Whether SGK moves to `0.2.3` is
   the user's (DatastoreKit `factory-defaults` U5); §5.9 says so. Not an issue.
8. **A transfer is not a resolution.** Each of the three moves to its board's §4 because the code is
   no longer SGK's, not because it is fixed. DatastoreKit's `docs/OPEN_ISSUES.md` §1.2 says the
   measurements "stay on SGK's boards", and points there. So each §4 entry keeps its text verbatim,
   and its added line says: transferred to DatastoreKit on 2026-10-12 by this campaign's prompt 09,
   **still open there**, indexed in DatastoreKit's `docs/OPEN_ISSUES.md` §1.2, this entry kept as the
   record of its measurement. Its heading's italic note gains "**transferred 2026-10-12 to
   DatastoreKit by dk-and-rpk-migration 09**". Append to the end of each board's §4 unless that board
   states an order. If a §3 lead-in on that board states a count of open issues, correct it; that is
   part of the move.
9. **The G2 drafts' form** (Z5). Each package board's §2 records a gate in a table cell:
   - **DatastoreKit's G2** cell begins `⬜` followed by dated italic notes. Its G1 reads `✅ <date>: …`.
     Draft the cell as `✅ 2026-10-12: holds …`, naming SGK's commits and runs, followed by the
     existing notes verbatim. Map each part to DatastoreKit's checklist,
     `docs/adoption/secondarygwkit.md` §9: its three parts (the 17 files deleted and the tag
     imported; the suites; the rehearsal's fingerprint).
   - **RayPoolKit's G2** cell reads "not yet (SGK adopts `v0.2.0`). Checklist: …". Its G1 reads
     `**holds** <date>, …`. Draft the cell in G1's form, keeping the checklist pointer, and map each
     part to RayPoolKit's checklist `docs/adoption/secondarygwkit.md` §8, items 1–5 (item 6 is the
     recording itself).
   - The sources are verification §2.10 and §3.10, plus 09's own final measurements, which name "this
     commit" until the orchestrator substitutes the SHA.
   - Each draft is a fenced block in verification §4, ready to paste. Neither is committed in its
     repository (C12).

**What moves on purpose:**
- **Modified:** the 18 files of Z2; `CLAUDE.md` (the test command only); the boards of
  `a3-v2-readiness`, `datastore-integrity` and `store-retirement` (the four entries only, and a §3
  count if one is stated); `docs/OPEN_ISSUES.md`; `docs/dk-and-rpk-migration-verification.md` (§4
  appended); README §2's table; this board; `prompts/INDEX.md`; the six probes (one statement each).
- **Added:** `measure/out/09-*`; the log.
- **Nothing else.** No test, no tool, no rehearsal script, no `requirements.txt`; nothing in
  DatastoreKit or RayPoolKit.
- **The index goes from 108 to 103**: three transferred, two closed, one narrowed and kept. By
  section: §1.15 6 → 5, §1.17 9 → 7, §1.20 4 → 2, §1.14 stays 4. "Twenty campaigns" stays: no
  section empties. §1.20's lead-in prose says what 09 closed.

**The facts, checked by the orchestrator on 2026-10-11/12 at `ede4062`.** Pass them on.

- **The tree and the machine.** Branch `handover-remedial`, `git status --short` and
  `git diff --cached` empty, this checkout alone in `git worktree list`. `ede4062` touches only
  `prompts/` since `f539015`. `RAY_ADDRESS` unset; `pgrep -lf 'gcs_server|raylet|ray::'` and the
  `ps` search for `SecondaryGWKit/venv` with `ray` find nothing. Load 12–27. 67 GB free on `var/`'s
  volume (93 % used).
- **The registry.** 29 runs: 27 finished, 2 `unknown` (the pre-registry `a3-pilot` pair). None
  `running`.
- **The venv.** `package_provenance()` gives 08's record: `datastorekit` 0.2.2 at
  `6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6` and `raypoolkit` 0.2.0 at
  `eaef0ce60e2fc354bb32f0ad576010e6a83c1976`, both `editable: false`, `requested_revision` the tags.
  `pip check` clean. **`pip freeze` equals `var/dk-and-rpk-migration-06/freeze-after.txt`** (`diff`
  empty).
- **§2.1's regex, the orchestrator's.** Over `git ls-files` outside `prompts/` and `docs/`, skipping
  lines that are import statements or hold `import_module(`:
  ```python
  mods = ["object", "contract", "replication", "shard_paths", "store_reader", "store_inventory",
          "SQL/schema", "SQL/ShardedPool", "SQL/Datastore", "SQL/ClientPool", "SQL/SerialPoolBroker",
          "SQL/ProfileAgent", "SQL/ObjectFactories/base"]
  alts = []
  for m in mods:
      alts.append(re.escape("Datastore/" + m) + r"(?:\.py)?\b")
      alts.append(re.escape("Datastore." + m.replace("/", ".")) + r"\b")
  for t in ["sharded_store", "shard_key_audit"]:
      alts += [re.escape("tools/" + t) + r"\b", re.escape("tools." + t) + r"\b"]
  alts += [r"ObjectFactories/base\b", r"RayTools"]
  pat = re.compile(r"(?<![\w./])(?:" + "|".join(alts) + ")")
  skip = re.compile(r"^\s*(?:from\s+\S+\s+import\b|import\s+\S)|import_module\(")
  ```
  The lookbehind `(?<![\w./])` is what keeps `datastorekit.tools.sharded_store` and
  `datastorekit/store_inventory.py` out. At `ede4062` it prints **exactly the board's 23 lines**, at
  the board's line numbers. The agent may use this or its own; either way the log gives it verbatim,
  its output before Z2 (23 lines) and after (none).
- **Z2's targets all exist** in the installed package or in SGK: `datastorekit.store_inventory`
  (`read_inventory`, `inventory_classes`), `datastorekit.shard_paths.resolve_shard_path`,
  `datastorekit.SQL.schema.drop_order`, `datastorekit/store_reader.py`, the method
  `Datastore._build_schema` in `datastorekit/SQL/Datastore.py` (`:400`), and
  `config.datastore.factories`. The seven `Datastore/tests` docstrings naming `._factories` belong to
  modules that take `factories as _factories` from `config.datastore` (six directly;
  `test_database_errors_reach_the_caller.py` through the modules it imports). `RunRegistry/stores.py:114`'s
  claim stays true under the new name: in a fresh `python -I`, `import RunRegistry.stores` loads
  neither `ray`, `sqlalchemy` nor `datastorekit.store_inventory`.
- **The six probes**, at `ede4062`, each fail on its one `_factories` statement:
  `docs/datastore-integrity-audit/fk_sweep_probe.py:8`, `lookup_keys/common.py:18`,
  `lookup_keys/probe_03_column_existence.py:25` (with `# noqa: E402`, which stays),
  `tq_fk_probe.py:8`; `docs/tolerance-convergence/inventory.py:196` and `order_audit.py:1253` (both
  nested, indented). `config.datastore.factories` has the old `_factories`'s shape, a dict from table
  name to factory (`config/datastore.py:91`; `Datastore/SQL/Datastore.py:89` at `19f07ee^`).
  `inventory.py:1315`'s string naming `Datastore/SQL/Datastore.py`'s `_factories` is a dated record
  in `docs/`, and Z7 leaves it.
- **The import check.** The orchestrator's script, run as `./venv/bin/python -I imports_only.py .
  <file>` from the repository root:
  ```python
  """Execute every absolute import statement of a Python file, and nothing else."""
  import ast, sys
  from pathlib import Path
  root, path = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
  src = path.read_text()
  sys.path[:0] = [str(root), str(path.parent)]
  nodes = [n for n in ast.walk(ast.parse(src)) if isinstance(n, (ast.Import, ast.ImportFrom))
           and not (isinstance(n, ast.ImportFrom) and n.level)]
  nodes.sort(key=lambda n: n.lineno)
  ns = {"__name__": "imports_only", "__file__": str(path)}
  for n in nodes:
      try:
          exec(compile(ast.Module([n], []), str(path), "exec"), ns)
      except Exception as e:
          print(f"{path.relative_to(root)}:{n.lineno}: {ast.unparse(n)}  ->  {type(e).__name__}: {e}")
          sys.exit(1)
  print(f"{path.relative_to(root)}: {len(nodes)} import statements, all OK")
  ```
  At `ede4062` each of the six exits 1 on its `_factories` line with `ImportError: cannot import name
  '_factories' from 'datastorekit.SQL.Datastore'`. With the repair applied to the source in memory,
  each prints `all OK`: **4, 6, 6, 4, 10 and 25** statements (in the order above). With
  `datastorekit.SQL.Datastore._factories` patched in for the prediction only, the six dependents
  print `all OK` at **14, 6, 9, 7, 14 and 7**. `git status` stayed clean.
- **The inherited entries**, at `ede4062`:
  - `prompts/a3-v2-readiness/IMPLEMENTATION_STATE.md` §3.2: `[00-a-new-stores-first-open-…]` at
    `:1353`, `[00-a-drop-action-…]` at `:1371`. Index rows `docs/OPEN_ISSUES.md:834-835` (§1.17).
  - `prompts/datastore-integrity/IMPLEMENTATION_STATE.md` §3.2: `[00-the-inventory-cannot-see-a-serial-split]`
    at `:1906`. Index row `:786` (§1.15).
  - `prompts/store-retirement/IMPLEMENTATION_STATE.md` §3: `[01-cross-filesystem-move-advice-…]` at
    `:525`; its §3 lead-in says "Four remain open", which stays true. Index row `:742` (§1.14).
  - DatastoreKit's `docs/OPEN_ISSUES.md` §1.2 (`83068cb`) lists all four, with SGK's board for each.
    No stop condition of the prompt's §4 fires.
- **The measurements at `ede4062`**, each run from its package's root with its own venv:
  - DatastoreKit's `measure_client_imports.py sgk --commit ede4062`: the **15** layer files absent;
    423 `.py` files read, 0 unparsed; correction 6's counts; 99 factory imports (15 in `docs/`).
    It equals `measure/out/04-datastorekit-measure_client_imports.md` but for the commit line and
    425 → 423 files read (07 deleted `RayTools/`'s two).
  - RayPoolKit's `measure_client_pools.py sgk --commit ede4062`: `RayTools/` absent, **0 imports**,
    **0 `import_module` calls**, 2 string literals, **54 calls, `constructs` 54**. It equals
    `measure/out/08-raypoolkit-measure_client_pools.md` but for 1495 tracked files (1474) and 788
    non-`.py` lines naming `RayTools` or `RayWorkPool` (651), the new ones in verification §3 and
    `prompts/`.
  - At the agent's commit both should equal these but for the commit and those record counts. Z7's
    six statements leave both scripts' counts unchanged: `datastorekit.SQL.Datastore` and
    `config.datastore` are neither SGK layer modules nor factory modules.
- **`CLAUDE.md`'s test command** is `:106-111`: one `discover` line for `ComputeTargets/tests` inside
  a bullet that goes on to the stand-in pattern. Z3 replaces the fenced block, and the bullet's lead
  if it must, and leaves `:110-111`'s sentence and the rest of the section as they are.
- **`prompts/INDEX.md`.** Header `:11`, "25 campaigns: 4 live, 1 seeded, 20 closed or complete", and
  `:12-15`, the sentence of dates on which counts were taken. This campaign's row is `:19`, "**live**
  — 9 of 9 written, 01–04 landed", "§1.20; 4". After 09: "3 live, 1 seeded, 21 closed or complete",
  the row closed in the form of its neighbours (e.g. `datastore-generic-followup`'s, "closed 3 / 3,
  2026-10-07 | 2026-10-06 → 2026-10-07"), and "§1.20; 2", the count taken 2026-10-12. Other
  campaigns' counts are dated snapshots, and are not 09's to retake.
- **Shell traps.** `stat` is GNU coreutils here. `grep` may wrap `ugrep`: use `git grep` or Python
  for any regex the log relies on. The shell is `zsh`: a bare word beginning `=` (`echo =====`) is
  command expansion and fails. Quote it.
- **Expected counts**, at `ede4062`, serially, load 12–23: AdaptiveLevin **32**, CosmologyModels
  **39**, LiouvilleGreen **148** (1 skipped), RunRegistry **286**, ComputeTargets **693**, Datastore
  **320** (54 + 103 + 101 + 62), all OK. The wall-clock test passed in the root run. No count moves
  at 09.
- **The index:** 108 now; **103** after.

**What the review exists to establish.** All else is mechanics.
- **(V1) The sweep is prose only.** Every Z2 module `ast`-equal with docstrings removed; each changed
  line changes a path and nothing else; the regex prints nothing after and 23 lines before.
- **(V2) The probes get past their imports**, and nothing else in them changed.
- **(V3) The hand-over is right.** Three entries moved verbatim with their line; one narrowed with
  its two halves stated; each board diff only that; the index rows and counts right by counting.
- **(V4) The close is true.** The measurements at the agent's commit; the G2 drafts each mapping to
  its own checklist and naming the right commits and runs; R0–R2's table matching the run
  directories.
- **(V5) The records close the campaign.** The board, README §2, `prompts/INDEX.md`, `CLAUDE.md`'s
  test command; §5.9 naming every issue left open and every decision left to the user.
- **(V6) The six roots pass** at 08's counts.

**Conventions.** 08's, carried. All bind the agent, and the brief says so.
- **Staging and reporting.** Stage by explicit path only, and show `git diff --cached --name-only`
  before committing. Count, then paste. A log states only what the record shows.
- **The SHA.** The agent writes "this commit", and the orchestrator substitutes the SHA in its
  follow-up. The final measurements are taken at the agent's `HEAD` before it commits, which is the
  note's commit (code unchanged but for Z2's prose and Z7); §4 says so.
- **Suites.** README §7's command, output to a file in the agent's scratch directory, the verdict
  grepped from it. Each **in the foreground**, serially, within the turn. Datastore in the four
  groups. Under load, the four tests of README §7 are known to fail: re-run a failure alone before
  calling it real.
- **Scratch.** The agent's own subdirectory of the session scratchpad; anything that must outlive
  the session under `var/dk-and-rpk-migration-09/`. No scratch file anywhere else in the repository.
- **The venv.** The agent installs and uninstalls nothing.
- **DatastoreKit and RayPoolKit.** Read only. Their scripts are run from their roots with their own
  venvs; nothing there is changed, fetched or committed (README §5 rule 12).
- **No store is opened**, no build runs and no Ray starts (prompt §5).
- **Never touch a process you did not start.**
- **Stop conditions are real.** A stop leaves the tree as it is. It goes to the user through the
  orchestrator, and the agent resumes with `SendMessage`.

## 1. Before you dispatch

Done on 2026-10-11/12 at `ede4062`, before this note was committed. Repeat steps 1–4 at the
dispatch `HEAD` if anything lands in between.

1. **The tree.** Record the branch and `HEAD`.
   - `git status --short` and `git diff --cached` must be empty.
   - `git worktree list` must show this checkout alone.
   - `git log --oneline f539015..HEAD` must touch nothing outside `prompts/`.
2. **The registry.** `./venv/bin/python -m RunRegistry list` must show nothing `running`. Report
   any stale or unknown run, without acting on it.
3. **Ray and the machine.** No Ray process of this repository; `RAY_ADDRESS` unset. Record the load.
4. **The venv.** `package_provenance()` gives §0's record; `pip check` clean; `pip freeze` equals
   06's after-freeze.
5. **Baselines**, at `ede4062`, serially:

   | Root | Tests | Result | Load |
   |---|---|---|---|
   | AdaptiveLevin | 32 | OK | 18 |
   | CosmologyModels | 39 | OK | 18 |
   | LiouvilleGreen | 148 | OK (1 skipped) | 18 |
   | RunRegistry | 286 | OK | 21 |
   | ComputeTargets | 693 | OK | 23 |
   | Datastore | 320 = 54 + 103 + 101 + 62 | OK | 12–14 |

   Every count is 08's.
6. **The index.** `grep -c '^| \`\[' docs/OPEN_ISSUES.md` gives 108.
7. **The regex, the import check and both measurements**, as §0.
8. **The user's decisions** on Z7 and the move advice are taken (§0).

## 2. Dispatch

One fresh-context subagent, on **Sonnet**. Give it:
- the prompt;
- the campaign README and this board;
- `CLAUDE.md` (the open-issues rule and "Repository mechanics");
- verification §2.10 and §3.10, and §1–§3's headers;
- the four inherited entries on their boards, and DatastoreKit's `docs/OPEN_ISSUES.md` §1.2;
- both packages' checklists' acceptance (DatastoreKit `docs/adoption/secondarygwkit.md` §9,
  RayPoolKit's §8) and both extraction boards' §2;
- `HEAD`, the baselines and the index count;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad,
  and the import-check script as a file beside it. The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the very end**, after every check and the suites. Its log is
  `logs/09-close-the-migration.md`, in README §5.1's form.
- **The order.** The regex and the import check first, before any edit, to record their "before".
  Then Z2 and Z7; then the regex, the `ast` comparison and the import check again; then Z1, Z3, Z6;
  then Z4's measurements at `HEAD` with the working tree's changes in place (the scripts read
  `git show`, so at `HEAD` they measure the committed tree: say so, and that the orchestrator re-runs
  them at the agent's commit); then Z5; then the suites.
- **The files it may change** are §0's "What moves on purpose", nothing else.
- **Stop and ask** on any of the prompt's §4 conditions.

## 3. The review — ten checks

1. **Scope.** `git diff --name-status HEAD~1 HEAD`: `A` for the log and the `measure/out/09-*`
   files; `M` for exactly §0's list. No test, tool, rehearsal script or `requirements.txt`. In the
   verification document, §4 is appended and nothing above it moves; the header already names §4.
2. **(V1) Z2.** For each of the 18 files:
   - `git diff -U0 HEAD~1 HEAD` shows only the board's lines (re-wrapped neighbours only where a
     line passed 100 columns, each named in the log), and each `-`/`+` pair differs only by the path
     (word-diff it);
   - `ast.dump` with docstrings removed equal before and after, run here;
   - the regex at `HEAD~1` prints the 23, and at `HEAD` nothing.
3. **(V2) Z7.** Each of the six diffs is one statement, `from config.datastore import factories as
   _factories`, with its indentation (and `probe_03`'s `# noqa: E402`). The import check at `HEAD`:
   the six and the six dependents `all OK` at §0's statement counts; at `HEAD~1`, the six fail as
   before. No probe was run.
4. **(V3) The boards.** For each of the three inherited boards, `git diff -U0` holds only the entry's
   removal from §3, its verbatim insertion in §4 with the italic note and the transfer line
   (correction 8), and a §3 count if one was stated. For store-retirement, only the narrowing line
   and, if the agent changed it, nothing else in the entry. Compare each moved entry with
   `HEAD~1`'s, character for character but for the added lines.
5. **(V3) The index.** Count the rows: 103 overall; §1.14 4, §1.15 5, §1.17 7, §1.20 2. The header's
   count and date (2026-10-12) agree. The three transferred rows and the two closed rows are gone;
   the narrowed row's hook names SGK's half. §1.20's lead-in is right.
6. **(V4) The measurements.** Run both packages' scripts at the agent's commit yourself. Each equals
   its saved `measure/out/09-*` and §0's figures but for the commit and the record counts. `pip
   freeze` equals 06's after-freeze; `package_provenance()` is §0's.
7. **(V4) Verification §4.** The R0–R2 table against verification §1–§3 and the run directories'
   `status.json` (twelve runs: tree, packages, fingerprint, end state). Each G2 draft: in its board's
   form (correction 9), every checklist item mapped to a commit or run that holds it, and nothing
   claimed that §2.10, §3.10 or Z4 does not show.
8. **(V5) The records.** The board: status line closed, every §1 row ✅ with its commit, Z1–Z7 ✅,
   §2 unchanged but for any closing line, §3 holding only what stays open, §4 holding the two closed,
   and `### 5.9 After prompt 09 (2026-10-12)` listing every issue left open (this campaign's two;
   store-retirement's narrowed half; the three now DatastoreKit's) and the three decisions left to
   the user, plus `0.2.3` (correction 7). README §2's nine rows landed with commits. `prompts/INDEX.md`
   as §0. `CLAUDE.md`'s diff only the test command, six roots and four groups in README §7's form,
   with the ten-minute sentence.
9. **(V6) The suites and `black`.** All six roots, serially, at the commit, every count as §1 step 5.
   `black --check` on every Python file touched.
10. **The log.** README §5.1's sections; this note's corrections and the user's two decisions
    classified; the regex and the import check quoted with their before and after; Z1's reading with
    its lines.

## 4. After the review

- **Record it on this board** as a §5.9 *Orchestrator review of prompt 09* paragraph, in 05's, 07's
  and 08's form. Fix small residue in a follow-up commit of its own:
  - "this commit" → the SHA, on the board, in the log's header, in verification §4 and in the two G2
    drafts;
  - this note added to the board's list of notes, marked "used for 09".
- Anything the review finds that is not residue is opened on this board's §3 and in the index.
- **Report to the user:**
  - V1–V6, each in a line;
  - the hand-over: three transferred, one narrowed, with where each now stands;
  - the final measurements and counts;
  - the two G2 drafts' location, and that committing them in DatastoreKit and RayPoolKit is the
    user's (C12);
  - what is left to the user: the six stores (README §3), the two G2 commits, the `Datastore` rename,
    and whether to move to `datastorekit` 0.2.3.
