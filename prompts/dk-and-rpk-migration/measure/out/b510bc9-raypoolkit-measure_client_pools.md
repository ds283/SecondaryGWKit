# `RayWorkPool` in SecondaryGWKit (SGK) at `b510bc9`

- **Repository:** `/Users/ds283/Documents/Code/SecondaryGWKit`, read through `git` only.
- **Commit:** `b510bc90b66985532d28996ab4e5b13b0ff71709`.
- **Verdicts by:** `raypoolkit 0.2.0`, its own constructor.
- **Files:** 1432 tracked; 460 `.py` files outside `RayTools/` parsed; 0 did not parse; 0 binary, not searched; 0 datastores, not read.
- **`ray.is_initialized()` after the verdicts:** False.

## `RayTools/` at the commit

| File | Blob |
|---|---|
| `RayTools/RayWorkPool.py` | `b4666fa89abb26829fd585b881947d7ef5172cea` |
| `RayTools/__init__.py` | `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` |

## Totals

| Measure | Count |
|---|---|
| Imports of a `RayTools` module | 7 (module level 7, nested 0) |
| … naming a private name | 0 |
| String literals naming `RayTools` | 8 |
| `import_module` / `__import__` calls naming `RayTools` | 0 |
| `RayWorkPool(...)` calls | 54 |
| … no compute handler (`compute_handler=None`) | 44 |
| … compute handler given | 6 |
| … compute handler left at its default | 4 |
| … `store_results=True` | 39 |
| … with a `title` keyword | 54 (`title=None` 28, set 26) |
| … positional arguments | 2 at 54 |
| … with `*args` / `**kwargs` | 0 / 0 |
| Verdicts | `constructs` 54 |
| Calls with a flag / flags | 0 / 0 |
| Lines of non-`.py` files naming `RayTools` or `RayWorkPool` | 196 |
| Test modules naming `RayTools` or `RayWorkPool` | 7 |

## The handler keywords, by spelling

Each handler keyword as absent (—), `None` or set; a row per spelling, with its calls.

| `compute_handler` | `store_handler` | `persist_handler` | `available_handler` | `validation_handler` | `post_handler` | `label_builder` | Calls |
|---|---|---|---|---|---|---|---|
| None | None | None | None | None | None | None | 6 |
| None | None | None | None | None | — | None | 27 |
| None | None | None | None | None | set | None | 1 |
| None | None | None | — | — | — | — | 10 |
| — | — | — | — | None | — | set | 1 |
| — | — | — | — | — | — | — | 2 |
| — | — | — | — | set | — | set | 1 |
| set | — | — | — | None | set | None | 1 |
| set | — | — | — | set | None | set | 1 |
| set | — | — | — | set | — | set | 2 |
| set | — | — | — | set | set | set | 2 |

## Imports of a `RayTools` module (7)

| File:line | Scope | Statement | Private names |
|---|---|---|---|
| `extract_GkSource_data.py:37` | `(module)` | `from RayTools.RayWorkPool import RayWorkPool` | — |
| `extract_GkWKB_data.py:35` | `(module)` | `from RayTools.RayWorkPool import RayWorkPool` | — |
| `extract_Gk_data.py:32` | `(module)` | `from RayTools.RayWorkPool import RayWorkPool` | — |
| `extract_QuadSourceIntegral_data.py:42` | `(module)` | `from RayTools.RayWorkPool import RayWorkPool` | — |
| `extract_TkWKB_data.py:30` | `(module)` | `from RayTools.RayWorkPool import RayWorkPool` | — |
| `extract_tensor_source_data.py:31` | `(module)` | `from RayTools.RayWorkPool import RayWorkPool` | — |
| `main.py:63` | `(module)` | `from RayTools.RayWorkPool import RayWorkPool` | — |

## String literals naming `RayTools` (8)

```text
Datastore/tests/test_layer_is_generic.py:1 [(module)] "\nThe generic layer names nothing of the project, and imports nothing of it but two generic\nmodules (prompts/datastore-generic, prompt 09; README \u00a75 rule
Datastore/tests/test_layer_is_generic.py:70 [layer_files] "RayTools"
Datastore/tests/test_layer_is_generic.py:88 [(module)] "RayTools/RayWorkPool.py"
Datastore/tests/test_layer_is_generic.py:120 [(module)] "RayTools"
Datastore/tests/test_layer_is_generic.py:192 [import_allowed] "RayTools"
Datastore/tests/test_layer_is_generic.py:357 [TestTheLayerImportsNoProjectPackage.test_the_allow_list] "RayTools.RayWorkPool"
docs/tolerance-convergence/inventory.py:382 [(module)] "RayTools"
prompts/datastore-generic/orchestrator/measure-09/scripts/m3_make_mutations.py:33 [(module)] "    for path in (REPO_ROOT / \"RayTools\").glob(\"*.py\"):\n        found.add(path.relative_to(REPO_ROOT).as_posix())\n"
```

## `import_module` / `__import__` calls naming `RayTools` (0)

None.

## `RayWorkPool(...)` calls (54)

Each call at the line of its name. Handlers: absent (—), `None` or set. Sizes: the source text of each size keyword given. `results`: `store_results`. Verdict: `v0.2.0`'s constructor, given the classes.

| File:line | Scope | Pos. | `**` | compute | store | persist | available | validation | post | label | Sizes | `results` | Title | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `extract_GkSource_data.py:812` | `run_pipeline` | 2 | — | None | None | None | — | — | — | — | — | True | set | `constructs` |
| `extract_GkSource_data.py:825` | `run_pipeline` | 2 | — | None | None | None | — | — | — | — | — | True | set | `constructs` |
| `extract_GkSource_data.py:910` | `run_pipeline` | 2 | — | None | None | None | None | None | None | None | `create_batch_size=10`, `process_batch_size=10` | False | set | `constructs` |
| `extract_GkWKB_data.py:403` | `run_pipeline` | 2 | — | None | None | None | — | — | — | — | — | True | set | `constructs` |
| `extract_GkWKB_data.py:416` | `run_pipeline` | 2 | — | None | None | None | — | — | — | — | — | True | set | `constructs` |
| `extract_GkWKB_data.py:483` | `run_pipeline` | 2 | — | None | None | None | None | None | None | None | `create_batch_size=10`, `process_batch_size=10` | False | set | `constructs` |
| `extract_Gk_data.py:359` | `run_pipeline` | 2 | — | None | None | None | — | — | — | — | — | True | set | `constructs` |
| `extract_Gk_data.py:372` | `run_pipeline` | 2 | — | None | None | None | — | — | — | — | — | True | set | `constructs` |
| `extract_Gk_data.py:432` | `run_pipeline` | 2 | — | None | None | None | None | None | None | None | `create_batch_size=10`, `process_batch_size=10` | False | set | `constructs` |
| `extract_QuadSourceIntegral_data.py:1105` | `run_pipeline` | 2 | — | None | None | None | — | — | — | — | — | True | set | `constructs` |
| `extract_QuadSourceIntegral_data.py:1118` | `run_pipeline` | 2 | — | None | None | None | — | — | — | — | — | True | set | `constructs` |
| `extract_QuadSourceIntegral_data.py:1272` | `run_pipeline` | 2 | — | None | None | None | None | None | None | None | `create_batch_size=10`, `process_batch_size=10` | False | set | `constructs` |
| `extract_TkWKB_data.py:422` | `run_pipeline` | 2 | — | None | None | None | — | — | — | — | — | True | set | `constructs` |
| `extract_TkWKB_data.py:472` | `run_pipeline` | 2 | — | None | None | None | None | None | None | None | `create_batch_size=10`, `process_batch_size=10` | False | set | `constructs` |
| `extract_tensor_source_data.py:355` | `run_pipeline` | 2 | — | None | None | None | — | — | — | — | — | True | set | `constructs` |
| `extract_tensor_source_data.py:399` | `run_pipeline` | 2 | — | None | None | None | None | None | None | None | `create_batch_size=10`, `process_batch_size=10` | False | set | `constructs` |
| `main.py:1042` | `run_pipeline` | 2 | — | — | — | — | — | — | — | — | `create_batch_size=10`, `process_batch_size=10`, `max_task_queue=30` | True | set | `constructs` |
| `main.py:1057` | `run_pipeline` | 2 | — | — | — | — | — | — | — | — | `create_batch_size=10`, `process_batch_size=10`, `max_task_queue=30` | True | set | `constructs` |
| `main.py:1319` | `run_pipeline.build_Tk_numeric_work` | 2 | — | None | None | None | None | None | — | None | `create_batch_size=20`, `process_batch_size=20` | True | None | `constructs` |
| `main.py:1420` | `run_pipeline` | 2 | — | set | — | — | — | set | set | set | `create_batch_size=2`, `process_batch_size=50`, `max_task_queue=100` | False | set | `constructs` |
| `main.py:1483` | `run_pipeline.build_Tk_WKB_work` | 2 | — | None | None | None | None | None | — | None | `create_batch_size=20`, `process_batch_size=20` | True | None | `constructs` |
| `main.py:1550` | `run_pipeline.build_Tk_WKB_work` | 2 | — | None | None | None | None | None | — | None | `create_batch_size=20`, `process_batch_size=20` | True | None | `constructs` |
| `main.py:1634` | `run_pipeline` | 2 | — | set | — | — | — | set | — | set | `create_batch_size=2`, `process_batch_size=50`, `max_task_queue=100` | False | set | `constructs` |
| `main.py:1701` | `run_pipeline.build_tensor_source_work` | 2 | — | None | None | None | None | None | — | None | `create_batch_size=20`, `process_batch_size=20` | True | None | `constructs` |
| `main.py:1767` | `run_pipeline.build_tensor_source_work` | 2 | — | None | None | None | None | None | — | None | `create_batch_size=20`, `process_batch_size=20` | True | None | `constructs` |
| `main.py:1824` | `run_pipeline.build_tensor_source_work` | 2 | — | None | None | None | None | None | — | None | `create_batch_size=20`, `process_batch_size=20` | True | None | `constructs` |
| `main.py:1940` | `run_pipeline` | 2 | — | — | — | — | — | set | — | set | `create_batch_size=1`, `process_batch_size=20`, `max_task_queue=300` | False | set | `constructs` |
| `main.py:1993` | `run_pipeline.build_Gk_numeric_work` | 2 | — | None | None | None | None | None | — | None | `create_batch_size=20`, `process_batch_size=20` | True | None | `constructs` |
| `main.py:2115` | `run_pipeline` | 2 | — | set | — | — | — | set | set | set | `create_batch_size=2`, `process_batch_size=50`, `max_task_queue=100` | False | set | `constructs` |
| `main.py:2188` | `run_pipeline.build_Gk_WKB_work` | 2 | — | None | None | None | None | None | — | None | `create_batch_size=20`, `process_batch_size=20` | True | None | `constructs` |
| `main.py:2265` | `run_pipeline.build_Gk_WKB_work` | 2 | — | None | None | None | None | None | — | None | `create_batch_size=20`, `process_batch_size=20` | True | None | `constructs` |
| `main.py:2428` | `run_pipeline` | 2 | — | set | — | — | — | set | — | set | `create_batch_size=2`, `process_batch_size=50`, `max_task_queue=100` | False | set | `constructs` |
| `main.py:2502` | `run_pipeline.resolve_GkSource_parents` | 2 | — | None | None | None | None | None | — | None | `create_batch_size=5`, `process_batch_size=10` | True | None | `constructs` |
| `main.py:2584` | `run_pipeline.build_GkSource_batch` | 2 | — | None | None | None | None | None | — | None | `create_batch_size=20`, `process_batch_size=20` | True | None | `constructs` |
| `main.py:2659` | `run_pipeline.build_missing_GkSource` | 2 | — | None | None | None | None | None | — | None | `create_batch_size=5`, `process_batch_size=10` | True | None | `constructs` |
| `main.py:2926` | `run_pipeline.build_incomplete_GkSourcePolicy` | 2 | — | None | None | None | None | None | — | None | `create_batch_size=20`, `process_batch_size=20` | True | None | `constructs` |
| `main.py:2974` | `run_pipeline.build_incomplete_GkSourcePolicy` | 2 | — | None | None | None | None | None | set | None | `create_batch_size=20`, `process_batch_size=20` | False | None | `constructs` |
| `main.py:3034` | `run_pipeline.build_GkSourcePolicy_batch` | 2 | — | None | None | None | None | None | — | None | `create_batch_size=20`, `process_batch_size=20` | True | None | `constructs` |
| `main.py:3074` | `run_pipeline.build_GkSourcePolicy_batch` | 2 | — | None | None | None | None | None | — | None | `create_batch_size=20`, `process_batch_size=20` | True | None | `constructs` |
| `main.py:3163` | `run_pipeline.build_missing_GkSourcePolicy` | 2 | — | None | None | None | None | None | — | None | `create_batch_size=20`, `process_batch_size=20` | True | None | `constructs` |
| `main.py:3229` | `run_pipeline` | 2 | — | set | — | — | — | set | None | set | `create_batch_size=1`, `process_batch_size=50`, `max_task_queue=100` | False | set | `constructs` |
| `main.py:3247` | `run_pipeline` | 2 | — | set | — | — | — | None | set | None | `create_batch_size=1`, `process_batch_size=50`, `max_task_queue=100` | False | set | `constructs` |
| `main.py:3350` | `run_pipeline.resolve_QuadSourceIntegral_parents` | 2 | — | None | None | None | None | None | — | None | `create_batch_size=20`, `process_batch_size=20` | True | None | `constructs` |
| `main.py:3398` | `run_pipeline.resolve_QuadSourceIntegral_parents` | 2 | — | None | None | None | None | None | — | None | `create_batch_size=20`, `process_batch_size=20` | True | None | `constructs` |
| `main.py:3460` | `run_pipeline.resolve_QuadSourceIntegral_parents` | 2 | — | None | None | None | None | None | — | None | `create_batch_size=20`, `process_batch_size=20` | True | None | `constructs` |
| `main.py:3515` | `run_pipeline.resolve_QuadSourceIntegral_parents` | 2 | — | None | None | None | None | None | — | None | `create_batch_size=20`, `process_batch_size=20` | True | None | `constructs` |
| `main.py:3557` | `run_pipeline.resolve_QuadSourceIntegral_parents` | 2 | — | None | None | None | None | None | — | None | `create_batch_size=20`, `process_batch_size=20` | True | None | `constructs` |
| `main.py:3686` | `run_pipeline.build_QuadSourceIntegral_batch` | 2 | — | None | None | None | None | None | — | None | `create_batch_size=20`, `process_batch_size=20` | True | None | `constructs` |
| `main.py:3783` | `run_pipeline.build_QuadSourceIntegral_batch` | 2 | — | None | None | None | None | None | — | None | `create_batch_size=20`, `process_batch_size=20` | True | None | `constructs` |
| `main.py:3824` | `run_pipeline.build_QuadSourceIntegral_batch` | 2 | — | None | None | None | None | None | — | None | `create_batch_size=20`, `process_batch_size=20` | True | None | `constructs` |
| `main.py:3886` | `run_pipeline.build_QuadSourceIntegral_batch` | 2 | — | None | None | None | None | None | — | None | `create_batch_size=20`, `process_batch_size=20` | True | None | `constructs` |
| `main.py:3958` | `run_pipeline.build_QuadSourceIntegral_batch` | 2 | — | None | None | None | None | None | — | None | `create_batch_size=20`, `process_batch_size=20` | True | None | `constructs` |
| `main.py:3997` | `run_pipeline.build_QuadSourceIntegral_batch` | 2 | — | None | None | None | None | None | — | None | `create_batch_size=20`, `process_batch_size=20` | True | None | `constructs` |
| `main.py:4146` | `run_pipeline` | 2 | — | — | — | — | — | None | — | set | `create_batch_size=1`, `process_batch_size=100`, `max_task_queue=1000` | False | set | `constructs` |

## Calls with a compute path: the handlers given (10)

The source text of each handler keyword given, at each call whose compute handler is given or left at its default.

| File:line | `compute_handler` | `store_handler` | `persist_handler` | `available_handler` | `validation_handler` | `post_handler` | `label_builder` |
|---|---|---|---|---|---|---|---|
| `main.py:1042` | — | — | — | — | — | — | — |
| `main.py:1057` | — | — | — | — | — | — | — |
| `main.py:1420` | `compute_Tk_numeric_work` | — | — | — | `validate_Tk_numeric_work` | `lambda obj: record_unresolved_osc(Tk_unresolved_osc, obj)` | `build_Tk_numeric_work_label` |
| `main.py:1634` | `compute_Tk_numeric_work` | — | — | — | `validate_Tk_numeric_work` | — | `build_Tk_numeric_work_label` |
| `main.py:1940` | — | — | — | — | `validate_tensor_source_work` | — | `build_tensor_source_work_label` |
| `main.py:2115` | `compute_Gk_numeric_work` | — | — | — | `validate_Gk_numeric_work` | `lambda obj: record_unresolved_osc(Gk_unresolved_osc, obj)` | `build_Gk_numeric_work_label` |
| `main.py:2428` | `compute_Gk_WKB_work` | — | — | — | `validate_Gk_WKB_work` | — | `build_Gk_WKB_work_label` |
| `main.py:3229` | `compute_GkSource_work` | — | — | — | `validate_GkSource_work` | `None` | `build_GkSource_work_label` |
| `main.py:3247` | `compute_GkSourcePolicy_work` | — | — | — | `None` | `postprocess_GkSourcePolicy` | `None` |
| `main.py:4146` | — | — | — | — | `None` | — | `build_QuadSourceIntegral_label` |

## Flags (0, on 0 calls)

None.

## Notification keywords, by source text (16 calls; never passed)

| File:line | Keywords |
|---|---|
| `extract_GkSource_data.py:910` | `notify_batch_size=50`, `notify_time_interval=120` |
| `extract_GkWKB_data.py:483` | `notify_batch_size=50`, `notify_time_interval=120` |
| `extract_Gk_data.py:432` | `notify_batch_size=50`, `notify_time_interval=120` |
| `extract_QuadSourceIntegral_data.py:1272` | `notify_batch_size=50`, `notify_time_interval=120` |
| `extract_TkWKB_data.py:472` | `notify_batch_size=50`, `notify_time_interval=120` |
| `extract_tensor_source_data.py:399` | `notify_batch_size=50`, `notify_time_interval=120` |
| `main.py:1042` | `notify_batch_size=10`, `notify_min_time_interval=MIN_NOTIFY_INTERVAL` |
| `main.py:1057` | `notify_batch_size=10`, `notify_min_time_interval=MIN_NOTIFY_INTERVAL` |
| `main.py:1420` | `notify_batch_size=500`, `notify_min_time_interval=MIN_NOTIFY_INTERVAL` |
| `main.py:1634` | `notify_batch_size=500`, `notify_min_time_interval=MIN_NOTIFY_INTERVAL` |
| `main.py:1940` | `notify_min_time_interval=MIN_NOTIFY_INTERVAL` |
| `main.py:2115` | `notify_batch_size=2000`, `notify_min_time_interval=MIN_NOTIFY_INTERVAL` |
| `main.py:2428` | `notify_batch_size=2000`, `notify_min_time_interval=MIN_NOTIFY_INTERVAL` |
| `main.py:3229` | `notify_batch_size=2000`, `notify_min_time_interval=MIN_NOTIFY_INTERVAL` |
| `main.py:3247` | `notify_batch_size=2000`, `notify_min_time_interval=MIN_NOTIFY_INTERVAL` |
| `main.py:4146` | `notify_batch_size=2000`, `notify_min_time_interval=MIN_NOTIFY_INTERVAL` |

## Non-`.py` files naming `RayTools` or `RayWorkPool` (196 lines)

| Top directory | Lines | Files |
|---|---|---|
| `CLAUDE.md` | 1 | 1 |
| `docs/` | 39 | 10 |
| `prompts/` | 156 | 44 |

Each line outside the campaign records (`prompts/`, `.prompts/`), 40:

```text
CLAUDE.md:71: `RayTools/` files are not frozen either: they are not part of this extraction. SecondaryGWKit's
docs/a3-v2-readiness-audit.md:88: - **Helpers that make no `pool.` call** [read]: `RayWorkPool` with `compute_handler`,
docs/a3-v2-readiness-audit.md:90: only its `task_builder` and never stores (`RayTools/RayWorkPool.py`, the `lookup` branch);
docs/a3-v2-readiness-audit.md:232: measured. `RayWorkPool`'s `ray.wait` has no timeout (read: `RayTools/RayWorkPool.py`, the main
docs/a3-v2-readiness-audit.md:583: `KeyboardInterrupt` keeps its type; anything else is held until the wait ends, and `RayWorkPool`'s
docs/a3-v2-readiness-audit.md:584: wait has no timeout (S1, measured; RayWorkPool read). **So the handler must record the signal
docs/a3-v2-readiness-verification.md:536: File ".../RayTools/RayWorkPool.py", line 331, in run
docs/a3-v2-readiness/rehearsal/reader_rows.txt:10: File "/Users/ds283/Documents/Code/SecondaryGWKit/RayTools/RayWorkPool.py", line 331, in run
docs/backport-modules-audit.md:16: | `RayTools/RayWorkPool.py` | 558 lines | 580 | 590 |
docs/backport-modules-audit.md:48: - `RayWorkPool.py`: differs by exactly one change, present **only in `SI`** (item **E1**).
docs/backport-modules-audit.md:78: | **B4** | `RayWorkPool` task builder returning `None` | CPBH→SI | **Backport** | XS |
docs/backport-modules-audit.md:81: | **E1** | `store_handler` / `persist_handler` split in `RayWorkPool` | SI | **Backport with care** | M (35 call sites) |
docs/backport-modules-audit.md:220: ### B4. `RayWorkPool` does not tolerate a task builder returning `None`
docs/backport-modules-audit.md:222: - **SGWK location:** `RayTools/RayWorkPool.py:283-297` (the `list`/`tuple`/`set` vs. scalar dispatch)
docs/backport-modules-audit.md:256: ### E1. `RayWorkPool`: `store_handler` / `persist_handler` split
docs/backport-modules-audit.md:258: - **SGWK locations:** `RayTools/RayWorkPool.py:64-65` (`_default_store_handler`), `89-97`
docs/backport-modules-audit.md:284: `extract_TkWKB_data.py`, `extract_tensor_source_data.py`; 45 `RayWorkPool(...)` constructions in
docs/backport-modules-audit.md:429: - **D4.** `RayWorkPool` notification bookkeeping: `self._last_num_available_complete = self._num_store_complete`
docs/backport-modules-audit.md:430: (`SGWK/RayWorkPool.py:543`, `SI:575`) — copy-paste; should be `self._num_available_complete`.
docs/backport-modules-audit.md:456: `RayWorkPool` + `ClientPool`.
docs/backport-modules-audit.md:459: 5. **Commit 5 — `persist_handler` split (E1), if wanted.** `RayWorkPool` + all 35 `store_handler=None`
docs/backport-modules-verification.md:40: e81b145 Tidy up latent faults in ShardedPool, RayWorkPool and ClientPool (prompt 03)
docs/backport-modules-verification.md:44: 0ee16a8 Split RayWorkPool store_handler into store and persist handlers (prompt 05)
docs/backport-modules-verification.md:75: | `RayWorkPool(` | 45 | **45** |
docs/backport-modules-verification.md:110: | Byte-compile | `./venv/bin/python3 -m py_compile main.py extract_*.py Datastore/SQL/*.py Datastore/SQL/ObjectFactories/*.py RayTools/*.py config/sharding.py tools/*.py` | exit 0 |
docs/backport-modules-verification.md:112: | Import, for real | `import Datastore.SQL.ShardedPool`, `Datastore.SQL.Datastore`, `RayTools.RayWorkPool`, `config.sharding` | all import clean, **no `sys.modules` stub needed** — confirms `d6f4a43` generalises |
docs/backport-modules-verification.md:140: | 6 | Task builder returning `None` completes under `store_results=False`, raises under `store_results=True` | **RUN — PASS** | Against a real `RayWorkPool(pool, ...)` built from the live pool (prompt 03's own verification used a dummy `pool` stand-in): both branches behave exactly as B4 requires. |
docs/backport-modules-verification.md:141: | 7 | Default handlers still store a result correctly | **NOT RUN** | Needs a real compute-target object (`.available`/`.compute()`/`.store()` contract) carried through `RayWorkPool`'s compute→store→persist cycle. See §5. |
docs/backport-modules-verification.md:179: - **Check 7 (default handlers store a real compute result end-to-end).** `RayWorkPool`'s
docs/backport-modules-verification.md:183: for real needs one genuine compute-target object carried through a live `RayWorkPool`, which means
docs/backport-modules-verification.md:215: `Datastore`/`ShardedPool`/`RayWorkPool` test suite is future work, not something to bolt on here.
docs/backport-modules-verification.md:285: - `[01-shard-key-persistence]` and the `read_table`/`RayWorkPool`-negative-case half of
docs/datastore-generic-audit.md:242: `Datastore/shard_paths.py` and `RayTools/RayWorkPool.py`. The factories under OF are the client.
docs/datastore-generic-audit.md:244: infrastructure contract), `RayWorkPool.py` and `ProfileAgent.py` hold no project knowledge.
docs/gktk-remedial-verification.md:562: 16's wiring survives a real `RayWorkPool` with `store_results=False`.
docs/gktk-remedial-verification.md:704: 4. Prompt 16's per-$k$ unresolved-oscillation summary works in a real `RayWorkPool`: **16 printed
docs/one-loop-planning/resonance-scaffolding-v1/sigw-resonance-reconciliation.md:199: for every response `k`. It is looped, not vectorised over grid points (the `RayWorkPool` batches
docs/one-loop-planning/resonance-scaffolding-v1/sigw-resonance-reconciliation.md:287: **Vectorised or looped?** Looped over `(z_response, k, q, r)` work items through `RayWorkPool`, with
docs/source-remediation-verification.md:31: the oracle is valid.** A real `ShardedPool`/`RayWorkPool` run on a locally bootstrapped Ray
docs/tolerance-convergence/TOLERANCE-INVENTORY.md:563: Every keyword argument named `amplitude_rtol`, `atol`, `deriv_rtol`, `epsabs`, `epsrel`, `phase_atol`, `rtol`, `xtol` whose value is a numeric literal, in the production trees `AdaptiveLevin`, `ComputeTargets`, `CosmologyConcepts`, `CosmologyModels`, `Datastore`, `LiouvilleGreen`, `MetadataConcepts`, `Quadrature`, `RayTools`, `Units`, `config`, `utilities`, in `main.py` and in the `extract_*.py` readers. Test trees, `docs/` and `prompts/` are excluded: they measure the pipeline, they are not it.
```

## Test modules naming `RayTools` or `RayWorkPool` (7)

```text
ComputeTargets/tests/test_gksource_parent_set_main.py:218: """Stands in for RayWorkPool: runs the task builder over the items, in order, and keeps the
ComputeTargets/tests/test_gksource_parent_set_main.py:270: "RayWorkPool": _WorkPool,
ComputeTargets/tests/test_main_plumbing.py:432: """Both numeric RayWorkPool constructions pass a post_handler, and both are followed by a
ComputeTargets/tests/test_main_plumbing.py:449: and call.func.id == "RayWorkPool"
ComputeTargets/tests/test_main_plumbing.py:700: of the batch the enclosing `RayWorkPool` dispatches over. `unclassified` holds any site whose
ComputeTargets/tests/test_main_plumbing.py:730: # calls reached through a RayWorkPool task_builder: the tolerance is in the batch
ComputeTargets/tests/test_main_plumbing.py:736: and node.func.id == "RayWorkPool"
ComputeTargets/tests/test_main_plumbing.py:764: label = f"{scope_label}: RayWorkPool({batch_name}) line {call.lineno}"
ComputeTargets/tests/test_quadsource_integral_parent_main.py:615: """Stands in for RayWorkPool: runs the task builder over the items, in order, and keeps the
ComputeTargets/tests/test_quadsource_integral_parent_main.py:790: "RayWorkPool": _WorkPool,
ComputeTargets/tests/test_quadsource_policy_main.py:136: the ``key`` entry or keyword). "query" is a RayWorkPool dispatch over a batch of payload
ComputeTargets/tests/test_quadsource_policy_main.py:145: and node.func.id == "RayWorkPool"
ComputeTargets/tests/test_wkb_numeric_parent_main.py:77: RayWorkPool dispatch over a batch (an existence query or a stage lookup) and "construction" for
ComputeTargets/tests/test_wkb_numeric_parent_main.py:86: and node.func.id == "RayWorkPool"
ComputeTargets/tests/test_wkb_numeric_parent_main.py:114: """The RayWorkPool lines of `scope` dispatching a lookup of `cls`."""
ComputeTargets/tests/test_wkb_numeric_parent_main.py:120: and node.func.id == "RayWorkPool"
Datastore/tests/test_layer_is_generic.py:7: module of ``RayTools/``. ``layer_files()`` derives the set from the filesystem, not from
Datastore/tests/test_layer_is_generic.py:70: for path in (REPO_ROOT / "RayTools").glob("*.py"):
Datastore/tests/test_layer_is_generic.py:88: "RayTools/RayWorkPool.py",
Datastore/tests/test_layer_is_generic.py:120: _NOT_PROJECT_PACKAGES = {"Datastore", "RayTools", "config"}
Datastore/tests/test_layer_is_generic.py:182: # WallclockTimer and format_time (Datastore, ProfileAgent, RayWorkPool): a shared helper that
Datastore/tests/test_layer_is_generic.py:192: if root == "RayTools":
Datastore/tests/test_layer_is_generic.py:357: "RayTools.RayWorkPool",
RunRegistry/tests/test_pipeline_adoption.py:32: # Six lines in the vocabulary `RayWorkPool` prints, in the order it prints them: the banner from
```

## Files that do not parse (0)

None.
