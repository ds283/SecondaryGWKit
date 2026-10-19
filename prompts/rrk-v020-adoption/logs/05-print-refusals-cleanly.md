# Log — rrk-v020-adoption prompt 05, print the drivers' refusals cleanly

- **Subject:** [`05-print-refusals-cleanly.md`](../05-print-refusals-cleanly.md), board items P1–P3.
- **Commit:** `96a168e` (parent `affaf84`).
- **Result:** landed. P1–P3 pass; no stop condition fired. The one test failure, in `ComputeTargets`, is
  README §8's known wall-clock test, re-run alone twice and recorded.
- **Model:** Sonnet 5.5.
- **Date:** 2026-10-19.

**What 05 shows:** each driver prints a refusal as one `!!` line on stderr, exit status 1, no
traceback, and a failure keeps its traceback. The orchestrator's note for this prompt is cited as "the
orchestrator note for 05", by ruling number. The record is
[`docs/rrk-v020-adoption-verification.md`](../../../docs/rrk-v020-adoption-verification.md) §4.

## What shipped

- **`docs/handover/quadsource_atol_sweep.py`** and **`docs/gktk-remedial/scoped_pipeline_run.py`**: a
  `Refusal(RuntimeError)` class near the top; the refusal sites of prompt §2.2 raise it, messages
  unchanged; the three `runregistrykit.begin(` calls are wrapped in `try … except
  runregistrykit.ConfigError: raise … except RuntimeError as e: raise Refusal(str(e)) from e`; each
  `__main__` block is prompt §2.4's. `git diff` is the class, those sites, the wraps (the `begin`
  calls' lines are re-indented one level, which is the whole of the diff there) and the block.
- **`RunDrivers/tests/test_clean_refusals.py`** (new, 12 tests).
- **`docs/rrk-v020-adoption-verification.md`**: §4 appended. Nothing above it changed.
- **`prompts/rrk-v020-adoption/measure/out/05-roots.txt`**, this log, this board.
- **`docs/OPEN_ISSUES.md` is unchanged:** §3 gains no entry; the count stays 100.

**`Refusal` and `RuntimeError` raises, by function, after 05** (counted with `ast` before editing and
after; the test pins them):

| Driver | `raise Refusal(` | `raise RuntimeError(` |
|---|---|---|
| launcher | `run_build` 3, `assert_store_is_self_consistent` 1, `prepare` 2, `sweep` 1 (7) | `make_case_selector.select` 1, `substitute_main_py` 1, `run_child._filtered_build` 1, `run_build._filtered_build` 1, `sweep` 1 (5) |
| scoped driver | `main` 2, `register` 2 (4) | `main._filtered_build_model_list` 1, `main` 1 (2) |

The counts before editing were the note's ruling 3's, and the original line numbers were the prompt's
(10 and 5 `raise RuntimeError(` sites).

**SHA-256**, before and after:

| Script | Before | After |
|---|---|---|
| launcher | `2a98530369f530cfc6f33ae91ff78fb51d3cbc89dea56f8bcfb9cb9ee012676d` | `9b40e457f444db10d2b406b749e0b4391d53753e93c928fa431856b4c645e649` |
| scoped driver | `244ad70e7e0901c612bb01eb4f0fec10ee9bbd8e3bb3a17904024e111b6646df` | `d2e38510acf5cff09dc63219e5c59cead2eb28042ba58c770ab8abe90d972c83` |

**No refusal is raised after `register` or `begin` has taken over the output streams** (prompt §2.4).
Read from the code: in `run_build` both pre-checks and `assert_store_is_self_consistent` precede
`begin`, and `begin`'s own refusal is raised before its call returns, so the `sys.stdout` /
`sys.stderr` wrap (`RegisteredStream`) that follows is not yet installed; in `sweep` the same holds for
`begin`, and its other `Refusal` is `begin`'s; in `register` the two sites are before and at `begin`,
and the takeover follows it; in the scoped `main` both refusals are in the argument checks, before
`register`; in `prepare` and `run_child` nothing takes the streams over. The test at `register` also
asserts that nothing was taken over.

**The two `Refusal` classes never need to cross** (the note's §0, confirmed). The launcher loads
`scoped_pipeline_run.py` by path as `_scoped_pipeline_run`, and uses only `HEARTBEAT_MEANS`,
`BEAT_MIN_INTERVAL`, `StageTracker`, `RegisteredStream`, `SignalLatch`, `install_killed_handler`,
`finish_after_exec` and `terminal_state`: never `register` or `main`, the only functions of that module
that raise its `Refusal`. A test asserts the two classes are distinct and each a `RuntimeError`.

**`begin` at `v0.2.0` raises no other `RuntimeError`** (the note's ruling 1, repeated here, read with
`git -C /Users/ds283/Documents/Code/RunRegistryKit show v0.2.0:runregistrykit/{__init__,client,stores}.py`,
nothing changed or fetched there). From `begin` (`__init__.py:987`):
- `find_client_root` raises `ConfigError` (`client.py:84`, a `RuntimeError`) at `client.py:122-254`,
  before anything is created; `read_sidecar` (`stores.py:691`) "never raises on bad content", but a
  `ConfigError` "comes through" (its docstring);
- `RuntimeError` at `:1102`, a retired store, and `:1119`, a store a run that has not finished names,
  each ending `No run directory was created`: these are the refusals;
- `_running_runs_naming`'s own `RuntimeError` (`stores.py:1014`, the runs root is not a directory) is
  unreachable from `begin`, which calls it only `if os.path.isdir(runs_root)`;
- `FileExistsError` (`:1131`) and the `OSError`s of `os.makedirs` are not `RuntimeError`s;
- `_refuse_if_in_use` (`stores.py:1098`) and the sidecar writers' `RuntimeError`s are not called from
  `begin`; `this_host`, `package_provenance`, `git_provenance` and `write_json_atomic` raise none.
  Nothing else raises a `RuntimeError` from `begin`.

**What a log shows now** (the note's §0, not a deviation): a sweep's `--child` that meets
`assert_store_is_self_consistent`'s refusal prints one `!!` line and exits 1, where it printed a
traceback; the parent's "child … exited 1" `RuntimeError` is unchanged.

## Deviations from the prompt

Each of the note's rulings is logged as the classification it gives.

1. **`begin`'s three calls catch `runregistrykit.ConfigError` first and re-raise it**, before `except
   RuntimeError`. `ConfigError` is a `RuntimeError` at `v0.2.0`, and prompt §2.3's bare `except
   RuntimeError` would have made it a `Refusal`, against the same paragraph's "a `ConfigError` keeps
   its traceback". Settled by the orchestrator note for 05, ruling 1. `STRUCTURALLY REQUIRED`.
2. **Test 5 gains a `ConfigError` case, and there is a seventh breakage** (the orchestrator note for 05,
   ruling 1); test 2 pins the handler order at each `begin` call (ruling 3). `STRUCTURALLY REQUIRED`.
3. **Test 3 asserts the strict form.** The note's ruling 2 allows import-time warnings before the `!!`
   line only if one is in fact seen. None is seen on this machine: each subprocess's whole stderr is the
   one `!!` line. The test asserts that stderr is exactly one line, beginning `!! `. No relaxation was
   needed (the orchestrator note for 05, ruling 2). No classification: nothing departed.
4. **Test 2 counts by the innermost enclosing `def`, by dotted path**, as ruling 3 gives, and every count
   found in the code before editing was the note's. `IMPLEMENTATION CHOICE`.
5. **Test 4 at `register` subclasses `HardKilledPredecessorTestCase`**, importing that class alone, and
   leaves the child run alive (`a_child_run`), reaching `begin`'s `:1119` refusal through a live run
   (ruling 4). It asserts `Refusal`, `begin`'s message, a `__cause__` that is a plain `RuntimeError`
   equal in text, an unchanged runs root and nothing taken over. `IMPLEMENTATION CHOICE`.
6. **Test 4 at `run_build` and `sweep` loads the launcher fresh** with `test_quadsource_atol_sweep_prepare`'s
   `_load_script`, and takes `args` from the launcher's own `main()` with the mode functions stubbed (as
   `test_prune_on_resume.parsed_arguments` does, extended to `sweep`), then calls the real function. No
   parser function is added to a driver. `import ray` runs in the test process; `ray.is_initialized()`
   is asserted false after each, and `subprocess.Popen` is patched and asserted not called
   (ruling 5). `IMPLEMENTATION CHOICE`.
7. **Black reformatted the class's docstring** (the closing quotes on their own line) and left the wrapped
   `begin` calls' comment lines longer than 100 columns where they were already so; both are black's own.
   `IMPLEMENTATION CHOICE`.

## Verification performed

1. **Counts before and after** are in the table above; the ast test asserts them.
2. **The new module**: 12 tests, `OK`. They are `TestTheClass` (2), `TestTheSites` (2),
   `TestTheCleanPrint` (2), `TestBeginsRefusalAtRegister` (1), `TestBeginsRefusalAtTheLauncher` (2),
   `TestAFailureKeepsItsType` (3). The clean print, by hand, from the repository root: the launcher on
   `--build --database /etc/hosts --atol 1e-32 --rtol 1e-7` and the scoped driver on `… -- --database
   /etc/hosts` each print `!! <message>` on stderr, an empty stdout, exit 1.
3. **The seven roots** (`measure/out/05-roots.txt`; load is the one-minute average at the start of each):

   | Root | Ran | Result | Load (1 m) | Against |
   |---|---|---|---|---|
   | `AdaptiveLevin/tests` | 32 | OK | 42.4 | 02's 32 |
   | `CosmologyModels/tests` | 39 | OK | 42.4 | 39 |
   | `ObjectFactories/tests` | 279 | OK | 25.5 | 279 |
   | `tools/tests` | 91 | OK | 34.7 | 91 |
   | `LiouvilleGreen/tests` | 148 | OK (skipped=1) | 35.6 | 148 (1 skipped) |
   | `RunDrivers/tests` | 115 | OK | 40.8 | 103 + 12 |
   | `ComputeTargets/tests` | 700 | FAILED (failures=1) | 27.0 | 700 |

   ComputeTargets' one failure is README §8's `test_tk_wkb_phase.TestCost.test_wall_time_per_object`
   (`0.0543 not less than or equal to 0.05` in the full run). Re-run alone twice, at load 51.1 and 53.2:
   `0.1151` and `0.1045`, "not less than or equal to 0.06"; as at the baselines, and as 04's ruling 9. No
   other test failed.
4. **Every existing test passes unmodified**: `git diff --stat` shows no test file but the new one, and
   the 103 tests of `RunDrivers` that existed are among the 115 that pass, `test_quadsource_atol_sweep_prepare`,
   `test_build_rehearsal_scope`, `test_hard_killed_predecessor` and `test_pipeline_adoption` among them.
5. **`black --check`** is clean on the three `.py` files.
6. **No Ray was started** (`ray.init` is never called; the tests assert `ray.is_initialized()` is false),
   nothing under `var/` was written, no `begin` ran outside a test's temporary runs root, the rehearsal
   lock was not taken, no `runregistrykit end` or store command ran.

## The deliberate-breakage record

Each was applied in the tree, the whole `RunDrivers` root run (115 tests), the diff recorded, and the
tree restored (`cmp` against the saved copies is equal; the final `git diff` is 05's own change). Output
saved under the agent's scratch directory.

**1. The launcher's `__main__` calls `main()` bare** → `FAILED (failures=1)`:
`TestTheCleanPrint.test_the_launcher_prints_its_refusal_cleanly`.

```diff
 if __name__ == "__main__":
-    try:
-        main()
-    except Refusal as exc:
-        sys.stdout.flush()
-        print(f"!! {exc}", file=sys.stderr)
-        sys.exit(1)
+    main()
```

**2. `Refusal` subclasses `Exception`**, in both drivers → `FAILED (failures=3, errors=12)`:
- `test_clean_refusals`: `TestTheClass.test_refusal_is_a_runtime_error_in_each_driver` (both drivers),
  `TestTheClass.test_the_two_classes_are_distinct`, and `TestBeginsRefusalAtRegister.test_a_live_run_on_the_store_is_a_refusal_with_begins_message` (error);
- `test_hard_killed_predecessor`: `TestTheRefusal.test_register_meets_the_refusal_and_takes_nothing_over`,
  `TestEndThenTheResume.test_end_records_the_run_lost_and_the_second_registration_is_accepted` (errors);
- `test_quadsource_atol_sweep_prepare.SweepPrepareTestCase`: `test_a_running_run_in_the_given_runs_root_blocks_the_copy`,
  `test_a_tombstone_at_the_sweep_name_is_never_overwritten`, `test_an_existing_sweep_store_is_never_replaced`,
  `test_the_check_refuses_a_missing_serial`, `…_a_renamed_record`, `…_an_absolute_record` (two subtests),
  `…_an_extra_serial`, `…_two_swapped_serials` (errors).

```diff
-class Refusal(RuntimeError):
+class Refusal(Exception):
```
(in `quadsource_atol_sweep.py` and in `scoped_pipeline_run.py`)

**3. `run_build` lets `begin`'s `RuntimeError` through** → `FAILED (failures=2, errors=1)`:
`TestBeginsRefusalAtTheLauncher.test_run_build_makes_begins_refusal_a_refusal` (error),
`TestTheSites.test_each_begin_call_has_the_two_handlers_in_order` and
`TestTheSites.test_the_raise_sites_are_the_expected_ones` (launcher).

```diff
         except runregistrykit.ConfigError:
             raise
-        except RuntimeError as e:
-            raise Refusal(str(e)) from e
```

**4. `register` lets `begin`'s `RuntimeError` through** → `FAILED (failures=3)`:
`TestBeginsRefusalAtRegister.test_a_live_run_on_the_store_is_a_refusal_with_begins_message`,
`TestTheSites.test_each_begin_call_has_the_two_handlers_in_order` and
`TestTheSites.test_the_raise_sites_are_the_expected_ones` (scoped driver).

```diff
     except runregistrykit.ConfigError:
         raise
-    except RuntimeError as e:
-        raise Refusal(str(e)) from e
```

**5. The model filter's `raise RuntimeError` becomes `raise Refusal`** → `FAILED (failures=1)`:
`TestTheSites.test_the_raise_sites_are_the_expected_ones` (launcher).

```diff
-            raise RuntimeError("quadsource_atol_sweep --build: no LambdaCDM model")
+            raise Refusal("quadsource_atol_sweep --build: no LambdaCDM model")
```

**6. The `__main__` block exits 2** → `FAILED (failures=1)`:
`TestTheCleanPrint.test_the_launcher_prints_its_refusal_cleanly`.

```diff
-        sys.exit(1)
+        sys.exit(2)
```

**7. `run_build` loses its `ConfigError` clause** (the note's ruling 1) → `FAILED (failures=1, errors=1)`:
`TestAFailureKeepsItsType.test_a_config_error_from_begin_is_not_a_refusal` (error: the `ConfigError`
became a `Refusal`) and `TestTheSites.test_each_begin_call_has_the_two_handlers_in_order` (launcher).

```diff
-        except runregistrykit.ConfigError:
-            raise
         except RuntimeError as e:
             raise Refusal(str(e)) from e
```

Breakage 2's failures in the existing modules are what it proves: the subclassing matters
(the orchestrator note for 05, ruling 6). Not a deviation.

## Observations not acted on

1. **`prepare` turns any `RuntimeError` of `copy_store` into a `Refusal`**, as prompt §2.2 says. That
   includes a `ConfigError` (`copy_store` locates the client root), whose traceback is then lost. The
   prompt asks for the re-raise as written; a `ConfigError` there is not a refusal by ruling 1's
   reasoning. Not changed; not opened as an issue, since `prepare` is run by a person at a terminal and
   the message names the cause.
2. **`run_build`'s `import ray` precedes the pre-checks**, so that a refused `--build` imports Ray (a few
   seconds, nothing started). Left, as the prompt says (§2.5).
3. **The wall-clock test keeps failing under the machine's I/O load** (04's log, observation 2),
   at load 27–53 here, alone as well, as at the baselines. Not 05's.
4. **The scoped driver's `begin` call keeps its long comment lines** inside the new `try`; black leaves
   comments alone and three of them now run to 101 or 102 columns. Left.

## State handed to the next prompt

There is none: 05 is the reopened campaign's last prompt. The orchestrator's review should check:
- **"this commit"** wherever 05's hash belongs: this log's header, this board (§1 row 05 and P1–P3,
  §5.6) and README §2 if it names it. The verification document's §4 names no hash.
- **The pin**: the test module pins the sites and the handler order, so a new refusal fails
  `test_the_raise_sites_are_the_expected_ones` until the test is changed with it.
- **For the user:** the launcher's new SHA-256 is verification §4's; A3 v2's manifests will carry it.
- **The registry and the lock** were not touched; no Ray process of this repository was started.
