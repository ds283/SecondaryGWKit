# Orchestrator — prompt 07, adopt `raypoolkit`

Read [`../README.md`](../README.md) first: §0.1, §1, §2.2 (07's charter), §4, §4.2, §4.3, §5, §6.2
(C7, C8, C10) and §7. Then read [`prompt-04.md`](prompt-04.md), the note for the other swap, whose
conventions, dispatch and review this note carries over except where §0 below says otherwise, and
this board's *Orchestrator review of prompt 04* and *of prompt 06*.

**You do not write code.** You may:
- run RayPoolKit's measuring script from its root, with its venv, at any SGK commit;
- make scratch commit objects with `git commit-tree` from a **temporary index** in the session
  scratchpad, never the checkout's index, and never move a ref;
- run the suites, replay the log's breakage diffs with `git apply`, and revert them;
- run in-process checks from the session scratchpad, never committing one;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`07-adopt-raypoolkit.md`](../07-adopt-raypoolkit.md)
**Board items:** B1–B3 · **Closes:** nothing · **Narrows:** nothing · **Opens:** nothing expected
· **Model:** Sonnet, as the prompt recommends.

**Gate:**
- 06 landed (`b6cc119`) and was reviewed (`cd03ec4`). The tree is clean.
- `HEAD` is the commit that adds this note, or a later one that touches no code.
  - "Code" here means anything outside `prompts/`.
  - If a later commit touches code, re-measure §0's figures at the new `HEAD` and hand the agent
    the corrections as an addendum.
- `raypoolkit` 0.2.0 is installed from the pin at `eaef0ce…`, and `datastorekit` 0.2.2 at
  `6d8ee9ae…`, neither editable; `pip check` clean.
- Nothing is `running` in the registry, and no Ray process of this repository is up.

## 0. What makes this prompt unusual

**It is 04 for the other package, and much smaller.** Two files go, seven identical lines change,
and two pieces of prose are corrected. Nothing that runs changes but the module `RayWorkPool` is
loaded from, and RayPoolKit's script judges every one of the 54 constructions by the package's own
constructor. So the review has 04's three halves, each short:
- **the swap is exact**: the seven hunks are one line each, and the two prose edits are the only
  other changes;
- **nothing was missed**: the script at the commit finds 0 imports and 0 `import_module` calls
  naming `RayTools`, and `RayTools` cannot be imported;
- **every pool still constructs, and the pinned output still bites**: 54 `constructs`, the five
  `ast`-matching modules and `test_pipeline_adoption` pass, and the breakages are caught.

What the swap does to a build, `main.py:3247` above all, is 08's to measure (C8). 07 runs none.

**Corrections to the prompt.** The orchestrator checked the prompt against the tree at `cd03ec4`
and against both installed packages. Each correction below governs, and the agent logs each as an
`IMPLEMENTATION CHOICE` citing this note, or `STRUCTURALLY REQUIRED` where the prompt cannot be
followed as written.

1. **`test_pipeline_adoption.py`'s line numbers** (prompt "Read first" 6 says `:31-37`; B3.2 says
   `:31-33`). At `HEAD` the comment is **`:32-37`**, `A3_OUTPUT` is `:38-44`, and its banner is
   `:39`. The sentence B3.2 rewrites is the first, **`:32-33`** ("Six lines … (:566)."), as README
   C10 says. The rest of the comment, `:33-37` from "They are transcribed", is not B3.2's (below).
2. **Checklist item 5 names files 04 deleted.** Its trace of the persisted object cites
   `Datastore/SQL/ShardedPool.py:3328` and `:3418`, `Datastore/SQL/Datastore.py:598`, and
   `Datastore/object.py`, SGK's copies at `b510bc9`. Check the trace in the installed `datastorekit`
   instead, and say so in the log:
   - `datastorekit/SQL/ShardedPool.py:3441` is `object_store`; its sharded branch sends the object
     to its shard's actor by `self._shards[shard_id].object_store.remote(item)` at `:3531`;
   - `datastorekit/SQL/Datastore.py:661` is the actor's `object_store`, which returns
     `output_objects[0]` for one object (`:733`);
   - `datastorekit/object.py` defines no `__getstate__`, `__setstate__` or `__reduce__`; in the
     package only `replication.py` and `SQL/schema.py` do.

   SGK's side holds as the checklist says: `main.py:2871` (`postprocess_GkSourcePolicy`), `:3247`
   (the queue), `:3265` (the statistics banner); `config/sharding.py:32` (`GkSourcePolicyData` on
   `k`); the factory's `store`, `Datastore/SQL/ObjectFactories/GkSourcePolicyData.py:150-174`,
   setting `obj._my_id` at `:172` and returning `obj` at `:174`;
   `ComputeTargets/GkSourcePolicyData.py:519` (`_policy`) and `:573-574` (`_type`, `_quality`; the
   checklist says `:574-575`). In the package, the default handlers are
   `raypoolkit/RayWorkPool.py:79-84`, and the post handler is called with the persisted object at
   `:529-530`.
3. **The `prompts/` string literal** (§3.1) is
   `prompts/datastore-generic/orchestrator/measure-09/scripts/m3_make_mutations.py:33`. So the two
   are that line and `docs/tolerance-convergence/inventory.py:382`, at `HEAD` and after.

**The comment's other sentence — record, do not fix.** `test_pipeline_adoption.py:33-37` says the
block is "transcribed from `var/runs/a3-pilot/run.out`". It is not, verbatim: that file's progress
lines carry no UTC offset (`-- 2026-09-21 02:16:29 (4h 58m 12.4s running): 1/13 …`), the pool's
`datetime.now()` is naive so `%z` prints nothing (RayPoolKit checklist item 6), and neither stamp of
the block (`02:14:03+0100`, `07:36:58+0100`) is in `a3-pilot/run.out` or `a3-pilot-resume/run.out`.
The block still parses as the pool's vocabulary, which is what the test needs. B3.2 rewrites the
first sentence only, so the agent leaves `:33-37` and `A3_OUTPUT` byte for byte, records this under
"Observations not acted on", and **opens no issue for it**; the review decides what it is.

**`CLAUDE.md`'s section** "The datastore layer is `datastorekit`" is `CLAUDE.md:47-72` at `HEAD`:
**26 lines**, heading to the last line of its last paragraph (`:71-72`, the `RayTools/` paragraph,
which goes). So after B3.1 it is at most **29** lines, counted the same way. Nothing else in
`CLAUDE.md` moves.

**Breakage (i), tried in advance.** The orchestrator applied this to `main.py` at `HEAD`, ran the
five modules, and restored the file:
```diff
@@ -63,0 +64 @@ from RayTools.RayWorkPool import RayWorkPool
+import raypoolkit
@@ -3686 +3687 @@ def run_pipeline(
-        query_queue = RayWorkPool(
+        query_queue = raypoolkit.RayWorkPool(
```
`main.py:3686` is the `query_queue` of `build_QuadSourceIntegral_batch`. Six tests fail, none
errors: `test_quadsource_policy_main`'s `test_the_two_sites_are_found`,
`test_each_site_passes_QuadSource_policy_1pt5` and `test_the_GkSourcePolicy_is_still_passed_beside_it`,
and `test_main_plumbing`'s `ObjectGetToleranceWiringTestCase` `test_the_expected_sites_are_all_found`,
`test_every_object_get_has_a_classified_tolerance` and `test_the_already_decoupled_target_is_guarded`.
After 07 the context line is `from raypoolkit import RayWorkPool`, so the agent's diff differs from
this one there. It may use this site or another; at another site it says which tests fail.

**Breakage (iv), tried in advance.** `A3_OUTPUT`'s banner at `:39`, `** CALCULATE` → `* CALCULATE`:
`test_pipeline_adoption` gives 1 failure and 2 errors —
`TestStage.test_the_stage_is_the_line_that_had_to_be_reconstructed_by_hand` (`None != 'CALCULATE
QUADRATIC SOURCE INTEGRALS, 92.31% …'`), and the errors
`TestStage.test_the_stage_is_the_hand_written_manifests_own_field` (`TypeError`) and
`TestHeartbeatHonesty.test_the_stream_passes_everything_through_and_beats_on_a_new_stage`
(`KeyError: 'stage'`).

**What moves on purpose:**
- **Deleted (2):** `RayTools/RayWorkPool.py`, `RayTools/__init__.py`, by `git rm`.
- **Changed by one line each (7):** `main.py:63` and the six `extract_*.py` at the prompt's lines.
  `extract_common.py` does not import the pool and does not change.
- **Hand edits (2):** `CLAUDE.md`'s section (B3.1); `RunRegistry/tests/test_pipeline_adoption.py:32-33`
  (B3.2).
- **New:** `measure/out/07-raypoolkit-measure_client_pools.md`; the log.
- **Records:** this board (§1 row, B1–B3, `### 5.7 After prompt 07`), README §4's "07 ships" line.
  `docs/OPEN_ISSUES.md` only if an issue is opened.
- **Not touched:** every pool site and handler keyword; `main.py` but `:63`; every test but the one
  comment; `docs/` (`inventory.py:382` keeps `"RayTools"`, C10); `requirements.txt`; the venv;
  RayPoolKit and DatastoreKit.

**The facts, checked by the orchestrator on 2026-10-11 at `cd03ec4`.** The code outside `prompts/`
is `b6cc119`'s (`git diff --stat b6cc119 cd03ec4 -- . ':!prompts'` is empty). Pass them on.
- **The tree and the machine.** Branch `handover-remedial`, `git status --short` empty, this
  checkout alone in `git worktree list`. `RAY_ADDRESS` unset; `pgrep -lf
  'gcs_server|raylet|ray::'` finds nothing, and no process command line holds
  `SecondaryGWKit/venv`. Load 5–13 during the checks.
- **The registry.** `python -m RunRegistry list` shows 25 runs: 23 finished and 2 `unknown` (the
  pre-registry `a3-pilot` pair), none `running`.
- **The venv.** `raypoolkit` 0.2.0 and `datastorekit` 0.2.2 in this venv's `site-packages`;
  `pip check` clean.
- **The measurement at `HEAD`.** `measure_client_pools.py sgk --commit cd03ec4`, from RayPoolKit's
  root (clean on `main` at `c9b0e11`) with its venv, differs from
  `measure/out/b510bc9-raypoolkit-measure_client_pools.md` only by:
  - the commit, 1472 tracked files, 425 `.py` parsed;
  - `RayTools/RayWorkPool.py` blob `3c1d9bd5…` (04's U12 line; `b4666fa…` at `b510bc9`);
  - string literals **2** (8): `test_layer_is_generic.py`'s six went with that module at 04;
  - test modules naming `RayTools` or `RayWorkPool` **6** (7), for the same reason;
  - non-`.py` lines naming either, 441 (196): this campaign's records and 05's run records.

  Unchanged: **7 imports**, all module level, none private, at the prompt's seven lines;
  **0 `import_module` calls**; **54 calls, `constructs` 54**, with the handler-keyword table as at
  `b510bc9` (44 / 6 / 4 compute, 39 `store_results=True`, 26 titles set).
- **The swap, measured in advance.** A scratch commit of `HEAD` with the two files removed and the
  seven lines rewritten by `sed`, made from a temporary index (`2bdfdaa`, dangling, no ref), measures
  as `HEAD` but for: 1470 tracked files; no `RayTools/` blob rows; **0 imports** (`None.`). Strings
  2, `import_module` 0, **54 calls, every one `constructs`**, every other table byte-identical. The
  agent's commit adds its records, so its file counts differ by those; nothing else should.
- **The seven lines** are each exactly `from RayTools.RayWorkPool import RayWorkPool`, the only
  `RayTools` in their files. `black --check` is clean on `main.py`, the six importing `extract_*.py`
  and `test_pipeline_adoption.py`; the new line is shorter, so `black` changes nothing.
- **Outside `prompts/` and `docs/`, `RayTools` is named** only by the seven imports and
  `CLAUDE.md:71`. No `.py` comment or docstring outside them names `RayTools/` or a
  `RayWorkPool.py:` line but `test_pipeline_adoption.py:32-33`. So the prompt's "If you find other
  prose … record it" should find nothing outside `docs/` and `prompts/`.
- **The package's printed lines** are at `raypoolkit/RayWorkPool.py:206` (the banner
  `print(f"\n** {title}")`), `:588` (the first progress `print(msg)`; the second line is `:592`) and
  `:606` (the completion `print(`; the `Queue summary` line is `:609`). In SGK's copy they were
  `:166`, `:548` and `:566`.
- **The five `ast`-matching modules** at `HEAD`: `test_main_plumbing.py:445-450` and `:733-737`;
  `test_quadsource_policy_main.py:142-146`; `test_wkb_numeric_parent_main.py:83-87` and `:117-121`;
  `test_gksource_parent_set_main.py:270` and `test_quadsource_integral_parent_main.py:790`, which
  bind the name `RayWorkPool` to a stand-in. None executes `main.py`'s import line.
- **Entry points.** `main.py` and the six `extract_*.py` call `ray.init` at module scope:
  `py_compile` them, never import them.
- **On this Mac `stat` is GNU coreutils**, and `grep` in the agent's shell may wrap `ugrep`. Use
  `git grep` or Python for any regex the log relies on.
- **Expected counts after 07**, unchanged from dispatch (§1 step 5): Datastore 320
  (54 + 103 + 101 + 62), RunRegistry 286, ComputeTargets 693, AdaptiveLevin 32, CosmologyModels 39,
  LiouvilleGreen 148 (1 skipped).
- **The index:** 108, unchanged unless the agent opens an issue.

**What the review exists to establish.**
- **(E1) B1.** `git ls-tree HEAD RayTools` is empty; in a fresh `python -I`, `import RayTools` and
  `import RayTools.RayWorkPool` raise `ModuleNotFoundError`; `from raypoolkit import RayWorkPool`
  gives the class from `raypoolkit.RayWorkPool`; `ray.is_initialized()` false.
- **(E2) B2 is exact.** The commit's diff of each of the seven files is the one line, changed to
  `from raypoolkit import RayWorkPool` and nothing else.
- **(E3) B2 is complete.** The script at the commit gives the scratch figures above: 0 imports, 0
  `import_module` calls, the 2 strings, 54 `constructs`, the handler table unchanged.
- **(E4) B3.** `CLAUDE.md`'s section says what the prompt's B3.1 lists, in at most 29 lines, and
  nothing else in `CLAUDE.md` moves; `test_pipeline_adoption.py`'s diff is inside `:32-33` and its
  `ast` is otherwise unchanged.
- **(E5) The suites.** All six roots at their counts, the six named modules among them, `black`
  clean.
- **(E6) The breakages.** (i)–(iv) each fail what §0 and the prompt say.
- **(E7) The records.** The log, the board, README's line, and the index agree with the tree.

**Conventions.** Those of [`prompt-04.md`](prompt-04.md) §0 bind the agent, with these changes:
- **The venv.** The agent installs and uninstalls nothing.
- **RayPoolKit.** The agent runs `docs/adoption/measure_client_pools.py` from that root with that
  venv, redirecting its output to SGK's `var/dk-and-rpk-migration-07/` by absolute path. It changes,
  fetches into, checks out and commits nothing there, nor in DatastoreKit (README §5 rule 12).
- **Scratch commits.** For §3.1.2, the prompt's `git commit-tree` recipe on the final staged tree.
  For breakages (ii) and (iii), from a **temporary index** (`GIT_INDEX_FILE=<scratch>/idx git
  read-tree <tree>`, then `git update-index` the mutated blob), so the checkout's index never holds
  a mutation. No ref moves; nothing is amended.
- **Test ids before.** Before any edit, the agent saves `unittest -v` output of the five
  `ast`-matching modules and `test_pipeline_adoption` at `HEAD` under `var/dk-and-rpk-migration-07/`,
  and the same after; the comparison is of sorted ids with outcomes, and must be equal.
- **Ray.** The agent starts none. No test, import or check may initialise it.
- **The SHA.** The agent writes "this commit", and the orchestrator substitutes the SHA in its
  follow-up.
- **Suites.** README §7's command, output to a file in the agent's scratch directory, the verdict
  grepped from it. Each in the foreground, serially. Datastore in the four groups. Under load, the
  four tests of README §7 are known to fail: re-run a failure alone before calling it real.
  `test_tk_wkb_phase.TestCost.test_wall_time_per_object` failed in the root run at dispatch (§1
  step 5) and passed alone three times.
- **Stop conditions are the prompt's §5, unchanged.**

## 1. Before you dispatch

Done on 2026-10-11 at `cd03ec4`, before this note was committed. Repeat steps 1–4 at the dispatch
`HEAD` if anything lands in between.

1. **The tree.** Record the branch and `HEAD`.
   - `git status --short` and `git diff --cached` must be empty.
   - `git worktree list` must show this checkout alone.
   - `git log --oneline cd03ec4..HEAD` must touch no code; otherwise see the gate.
2. **The registry.** `./venv/bin/python -m RunRegistry list` must show nothing `running`.
3. **Ray.** `pgrep -lf 'gcs_server|raylet|ray::'`, and `ps -axo pid,ppid,command` for lines
   holding `SecondaryGWKit/venv`. Nothing of this repository.
4. **The venv.** `pip show raypoolkit datastorekit` gives 0.2.0 and 0.2.2 in this venv's
   `site-packages`; `pip check` is clean.
5. **Baselines**, at `cd03ec4`, serially, in the foreground:

   | Root | Tests | Result | Load |
   |---|---|---|---|
   | AdaptiveLevin | 32 | OK | 5 |
   | CosmologyModels | 39 | OK | 5 |
   | LiouvilleGreen | 148 | OK (1 skipped) | 6 |
   | RunRegistry | 286 | OK | 7 |
   | ComputeTargets | 693 | 1 failure: `test_wall_time_per_object` (0.0602 s against 0.06 s); alone, passed three times | 11 |
   | Datastore | 320 = 54 + 103 + 101 + 62 | OK | 5–6 |

   Every count is 04's.
6. **The index.** `grep -c '^| \`\[' docs/OPEN_ISSUES.md` gives 108.

## 2. Dispatch

One fresh-context subagent, on **Sonnet**. Give it:
- the prompt;
- the campaign README and this board;
- 04's log, which holds the shape of a swap's records, its §3.1 scratch measurement and its
  breakage diffs;
- `HEAD`, the baselines and the index count;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad.
  The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the end.** Its log is `logs/07-adopt-raypoolkit.md`, in README §5.1's form, with
  the prompt's §7 additions.
- **The files it may add, change or delete** are §0's "What moves on purpose", and nothing else. In
  particular, not `test_pipeline_adoption.py:33-37` or `A3_OUTPUT`, not `docs/`, not
  `orchestrator/`, not `measure/` but the one saved measurement, and nothing in RayPoolKit or
  DatastoreKit.
- **Order:** measure at `HEAD`; save the before-ids; `git rm` (B1); the seven lines (B2); the two
  prose edits (B3); `black --check`; stage and measure the staged tree; suites; §3.3; breakages;
  records; show `git diff --cached --name-only`; commit.
- **Stop and ask** on any of the prompt's §5 conditions.
- **Run the Datastore suite in four groups**, in the foreground.

## 3. The review — twelve checks

1. **Scope.** `git diff --name-status HEAD~1 HEAD`: `D` for the two `RayTools/` files; `A` for the
   saved measurement and the log; `M` for the seven importing files, `CLAUDE.md`,
   `test_pipeline_adoption.py`, this board and README, and `docs/OPEN_ISSUES.md` only with a new
   §3 entry. Nothing else.
2. **E2, the seven hunks.** `git diff -U0 HEAD~1 HEAD` on each of the seven files is one `-` line,
   `from RayTools.RayWorkPool import RayWorkPool`, and one `+` line,
   `from raypoolkit import RayWorkPool`, at the same line number. Independently: `sed` the seven
   `HEAD~1` blobs and compare with the committed blobs byte for byte.
3. **E3, by the script at the commit.** Run `measure_client_pools.py sgk --commit <commit>`. It
   gives §0's scratch figures; its call table and handler tables equal `HEAD~1`'s measurement
   line for line; it equals the saved `measure/out/07-…md` but for the commit lines and file counts.
   `git diff --stat <agent's scratch> <commit>` lists only the records. `git ls-tree <commit>
   RayTools` is empty.
4. **E1, the queue is the package's.** In a fresh `./venv/bin/python -I` with the repository on its
   path: `import RayTools` and `import RayTools.RayWorkPool` raise `ModuleNotFoundError`;
   `from raypoolkit import RayWorkPool` gives the class, `RayWorkPool.__module__` is
   `raypoolkit.RayWorkPool`, its file is the venv's; `ray.is_initialized()` false. `py_compile`
   `main.py` and the six `extract_*.py`. `RunRegistry list` runs.
5. **E4, `CLAUDE.md`.** The section's new title is the prompt's; it is at most 29 lines; it says
   what B3.1 lists (both packages, each reported to its own repository; the import and the bare
   name, with the reason; the `RayTools/` paragraph gone); and `git diff HEAD~1 HEAD -- CLAUDE.md`
   touches nothing outside `:47-72`. The test command is unchanged (09's).
6. **E4, `test_pipeline_adoption.py`.** The diff is inside the comment's first sentence. It names
   `raypoolkit/RayWorkPool.py:206`, `:588` and `:606`, each checked against the installed file, and
   says the block holds five lines: the banner and two progress notifications of two lines each,
   no completion line. `ast.dump` of the module equals `HEAD~1`'s (comments are not in the `ast`).
7. **E5, the suites.** All six roots, serially, at the commit, at §1 step 5's counts. The agent's
   before- and after-id lists of the six named modules are equal. `black --check` is clean on the
   eight touched `.py` files.
8. **E6, the breakages.** Replay (i)–(iv) from the log in `bash`, each from a clean tree:
   `git apply --check`, apply, run the named check (for (ii) and (iii), measure a temporary-index
   scratch commit), see the failure — or, for (iii)'s `test_main_plumbing`, the pass — `git apply
   -R`, and `git status` clean.
9. **Item 5, against the tree.** The log's row-by-row table says "nothing" for every row but
   `main.py:3247`, with the reason checked for each, and traces `:3247` through the installed
   `datastorekit` (§0 correction 2), not through the deleted copy.
10. **The registry and the machine.** `RunRegistry list` shows the same 25 runs, none `running`. No
    Ray process before or after. RayPoolKit's and DatastoreKit's checkouts are clean and at the
    same commits as before.
11. **E7, the records.**
    - The log: §0's corrections classified; both measurements against `b510bc9`'s appendix and §0's
      figures; the seven `-U0` hunks; item 5's table; the suites beside the baselines; the
      breakages as diffs, with what failed; the observation on `:33-37`; and, for 08, the commit,
      `package_provenance()`, and what R2 should expect to differ from R1 (the post handler at
      `main.py:3247`; the printed lines, which should not).
    - The board: the §1 row (✅, "this commit", the log), B1–B3 ✅, and
      `### 5.7 After prompt 07 (<date>)` naming the rollback: `git revert` alone (README §4.2).
    - README §4: the "07 ships" line.
    - The index at 108, unless the agent opened an issue, with the row, count and date in the
      same commit.
12. **The comment's provenance.** Decide what `test_pipeline_adoption.py:33-37` is (§0): residue
    for a follow-up, or an issue. It is not 07's diff either way.

## 4. After the review

- **Record it on this board** as a §5.7 *Orchestrator review of prompt 07* paragraph: what each
  check found, the measurement at the commit, the breakages, and anything handed on to 08. Fix
  small residue in a follow-up commit of its own. That includes:
  - "this commit" → the SHA, on the board and in the log's header;
  - `orchestrator/prompt-07.md` added to the board's list of notes, marked "used for 07".
- Anything the review finds that is not residue is opened on this board's §3 and in the index.
- **Report to the user:**
  - what went and what changed, with the seven hunks;
  - the measurement at the commit: 0 imports, 54 `constructs`;
  - the six roots' counts, and the breakages;
  - `CLAUDE.md`'s completed section;
  - the corrections this note made to the prompt, and the comment's provenance.
- **Then 08**, R2: §4.1's protocol at 07's tree, every manifest's `packages` naming both tags'
  commits. Its note is written after this review, and its L4 is the `GkSourcePolicyData` block
  against R1's, compared as sorted lines where only the order differs
  (`[02-the-policy-statistics-block-orders-its-lines-by-completion]`).
