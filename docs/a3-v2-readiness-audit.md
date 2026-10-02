# A3 v2 readiness audit — what 02, 03 and 04 rest on

**Date:** 2026-10-02 · **Tree:** `handover-remedial` at `5707851`, clean · **Author:** Claude Opus
5.5 · **Prompt:** [`prompts/a3-v2-readiness/00-audit-the-readiness-defects.md`](../prompts/a3-v2-readiness/00-audit-the-readiness-defects.md)
· **Status:** read-only. No production file, driver or test was changed. Every probe and its output
is under [`docs/a3-v2-readiness/`](a3-v2-readiness/). Under `var/`, one store was read: the A3
baseline, `mode=ro`, through `RunRegistry.stores.fingerprint_store(..., write=False)`, with the
SHA-256 of its five files and its sidecar unchanged afterwards (S4).

**Machine.** macOS (Darwin 25.5.0, arm64), 10 CPUs, Python 3.12.14, SQLite 3.53.4, **Ray 2.43.0**
(`ray-commit=ecdcdc6a…` in each `gcs_server`'s command line). Load averages between 5 and 16 while
the probes ran; each S probe prints the load before every variant.

**How each finding is marked.**
- **[measured]** — a committed probe reproduces it; the output file and the line are named.
- **[read]** — read in the source at `5707851`; the function is named. `Datastore/SQL/Datastore.py`
  and `Datastore/SQL/ShardedPool.py` are cited by function name only.
- **[inferred]** — neither; what it was inferred from is named. Nothing below is promoted from
  inferred to measured by argument.

**The probes.**

| Probe | Items | Needs Ray | Output |
|---|---|---|---|
| [`reader_writes_probe.py`](a3-v2-readiness/reader_writes_probe.py) | R2, R3 | no (stand-in shards) | [`.txt`](a3-v2-readiness/reader_writes_probe.txt) |
| [`readonly_open_probe.py`](a3-v2-readiness/readonly_open_probe.py) | M1, M2, V2 | no (stand-in shards) | [`.txt`](a3-v2-readiness/readonly_open_probe.txt) |
| [`version_window_probe.py`](a3-v2-readiness/version_window_probe.py) | V1 | no (stand-in shards) | [`.txt`](a3-v2-readiness/version_window_probe.txt) |
| [`sigterm_probe.py`](a3-v2-readiness/sigterm_probe.py) `S1`/`S2`/`S3` | S1, S2, S3 | **yes**, a local instance per variant | [`S1`](a3-v2-readiness/sigterm_probe_S1.txt), [`S2`](a3-v2-readiness/sigterm_probe_S2.txt), [`S3`](a3-v2-readiness/sigterm_probe_S3.txt) |
| [`fingerprint_time_probe.py`](a3-v2-readiness/fingerprint_time_probe.py) | S4 | no | [`.txt`](a3-v2-readiness/fingerprint_time_probe.txt) |

The stand-in probes import `Datastore/tests/standin_pool.py` (a real `ShardedPool`, the real
`Datastore` actor code, every factory and the broker, on SQLite files in a temporary directory) and
`Datastore/store_reader.py`; they copy neither. Their instruments are each file's SQLite **change
counter** (header bytes 24–27, which SQLite increments once per transaction that changes the file in
rollback-journal mode), the row count of every table in every file (read `mode=ro`), every file's
SHA-256, and a wrapper on `ShardedPool._replicated_write` that logs each call's operation, class and
whether its answer was inserted.

---

## R. Every write a reader makes

### R1. Every `pool.` call the six readers make, and what each reaches [read]

The six scripts share one structure (read at `5707851`): `ray.init` at module scope; `with
ShardedPool(..., prune_unvalidated=False, read_table_config=read_table_config) as pool:`;
`extract_common.resolve_run_selection(pool, args.run_label)`; `config.model_list.build_model_list(pool,
units)`; then `run_pipeline(model_data)` for **every** model the list returns. Replicated classes
are those in `config/sharding.py` `replicated_tables`: `version`, `store_tag`, `redshift`,
`wavenumber`, `wavenumber_exit_time`, `tolerance`, `LambdaCDM`, `QCD_Cosmology`,
`IntegrationSolver`, `BackgroundModel`, `BackgroundModelValue`, `GkSourcePolicy`,
`QuadSourcePolicy`.

Every `pool.object_get` on a replicated class goes through `ShardedPool._get_impl_replicated_table`
and so through `ShardedPool._replicated_write`, which commits a `replication_in_flight` row on the
primary before the controlling shard's call and deletes it afterwards, **whether or not anything is
inserted** (read: `_replicated_write`, steps 1 and 5). `pool.read_table` and every sharded lookup do
not (read: `ShardedPool.read_table`, `_get_impl_sharded_table`, `object_read_batch`).

| Call | Where (line at `5707851`) | Class | Replicated | Reaches | Inserts on a miss | Writes on a hit |
|---|---|---|---|---|---|---|
| `read_inventory(pool.primary)` | `extract_common.available_run_labels` L128 | (every class) | — | `store_reader.open_read_only`, `mode=ro` | no | no |
| `object_get("store_tag", label=Run_<run>)` | `extract_common.resolve_run_selection` L213, only when the store records a run | `store_tag` | yes | `sqla_store_tag_factory.build` | **yes** | in-flight record |
| `object_get("LambdaCDM", params, units)` | `config/model_list.build_model_list` L11–13 | `LambdaCDM` | yes | `sqla_LambdaCDM_factory.build` | **yes** | in-flight record |
| `object_get("QCD_Cosmology", params, units, max_z=1e20)` | `build_model_list` L15–17 | `QCD_Cosmology` | yes | `sqla_QCDCosmology_factory.build` | **yes** | in-flight record |
| `object_get(tolerance, tol=…)` ×4 | Gk L308–311, GkWKB L350–353, TkWKB L371–374, QSI L1053–1056 | `tolerance` | yes | `sqla_tolerance_factory.build` | **yes** | in-flight record |
| `object_get(tolerance, tol=…)` ×2 | GkSource L761–762, tensor L304–305 | `tolerance` | yes | as above | **yes** | in-flight record |
| `object_get(BackgroundModel, z_sample=None, tags=…)` | Gk L317, GkWKB L359, GkSource L768, TkWKB L380, tensor L311, QSI L1062 | `BackgroundModel` | yes | `sqla_BackgroundModelFactory.build` | no: an unpopulated object | in-flight record |
| `read_table("wavenumber", …)` ×2 (×1 in TkWKB, tensor) | Gk L340/342, GkWKB L384/386, GkSource L793/795, TkWKB L405, tensor L335, QSI L1086/1088 | `wavenumber` | yes | `sqla_wavenumber_factory.read_table`, one shard | no | no |
| `create_k_exit_work`: `object_get(wavenumber_exit_time, k, cosmology, atol, rtol)`, once per k read | Gk L346, GkWKB L390, GkSource L799, TkWKB L409, tensor L338, QSI L1092 | `wavenumber_exit_time` | yes | `sqla_wavenumber_exit_time_factory.build` | no: an unpopulated object | in-flight record |
| `read_table("redshift", model_proxy=…)` ×2 (×1 in GkWKB) | Gk L392/397, GkWKB L435, GkSource L841/846, QSI L1140/1145 | `redshift` | yes | `sqla_redshift_factory.read_table` | no | no |
| `object_get("GkSourcePolicy", …)` ×2 | GkSource L858/864, QSI L1161/1167 | `GkSourcePolicy` | yes | `sqla_GkSourcePolicy_factory.build` | **yes** | in-flight record |
| `find_QuadSource_policy`: `read_table("QuadSourcePolicy", …)` | QSI L1010 | `QuadSourcePolicy` | yes | `sqla_QuadSourcePolicy_factory.read_table` | no: refuses | no |
| `object_get("GkNumericIntegration", …)` | Gk L422, GkWKB L463 | sharded | no | `sqla_GkNumericIntegration_factory.build` | no | no |
| `object_get("GkWKBIntegration", labels_only=True, …)` | GkWKB L473 | sharded | no | `sqla_GkWKBIntegration_factory.build` | no | no |
| `object_get("GkSource", labels_only=True, …)` | GkSource L881, QSI L1212 | sharded | no | `sqla_GkSource_factory.build` | no | no |
| `object_get("GkSourcePolicyData", …)` | GkSource L893, QSI L1224 | sharded | no | `sqla_GkSourcePolicyData_factory.build` | no | no |
| `object_get("TkNumericIntegration", …)` | TkWKB L453 | sharded | no | `sqla_TkNumericIntegration_factory.build` | no | no |
| `object_get("TkWKBIntegration", labels_only=True, …)` | TkWKB L462 | sharded | no | `sqla_TkWKBIntegration_factory.build` | no | no |
| `object_get("QuadSource", labels_only=True, …)` | tensor L375, QSI L1234 | sharded | no | `sqla_QuadSource_factory.build` | no | no |
| `object_read_batch("QuadSourceIntegral", …)` | QSI L1193 | sharded | no | `sqla_QuadSourceIntegral_factory.read_batch` | no | no |
| `ShardedPool(...)` and its `__exit__` | each script's `with` | — | — | M1 below | — | M1 |

Abbreviations: Gk = `extract_Gk_data.py`, GkWKB = `extract_GkWKB_data.py`, GkSource =
`extract_GkSource_data.py`, TkWKB = `extract_TkWKB_data.py`, tensor = `extract_tensor_source_data.py`,
QSI = `extract_QuadSourceIntegral_data.py`.

- **Helpers that make no `pool.` call** [read]: `RayWorkPool` with `compute_handler`,
  `store_handler` and `persist_handler` all `None` (the exit-time queues and the plot queues) runs
  only its `task_builder` and never stores (`RayTools/RayWorkPool.py`, the `lookup` branch);
  `ModelProxy` and `GkSourceProxy` call `ray.put` only; `describe_background_generation`,
  `refuse_ambiguous_QuadSourceIntegrals` and every `@ray.remote` plot function take objects or
  refs, not the pool.
- **Sharded factories** [read]: no sharded class's `build` or `read_batch` contains an `inserter`
  call; every `inserter`/`inserters[...]` call in those modules is in `store`, `add_tag`, or a
  `*Value` factory's `_build_impl_serial`, which no reader reaches.
- **Two replicated factories write on a hit**, not only on a miss [read]: `sqla_wavenumber_factory.build`
  and `sqla_redshift_factory.build` update the `source`/`response` flags when the lookup asks for a
  flag the row lacks. No reader reaches either `build` (both are read with `read_table`), but a
  read-only pool must not rely on that: see M1.
- **The per-script set of replicated `object_get`s** is therefore: `store_tag` (one, if the store
  records a run), both cosmologies, then per model the script's tolerances (4 for Gk, GkWKB, TkWKB
  and QSI; 2 for GkSource and tensor), one `BackgroundModel`, one `wavenumber_exit_time` per k read
  (source and response k for Gk, GkWKB, GkSource and QSI; source only for TkWKB and tensor), and
  the two `GkSourcePolicy` rows (GkSource and QSI). QSI's is the widest; every other script's is a
  subset of its classes.

### R2. Measured on a stand-in: QSI's whole sequence [measured]

`reader_writes_probe.py` builds a store holding every row QSI asks for (its `## The full store`
line), transcribes QSI's `pool.` calls in order (with the line numbers above), and adds the four
sharded lookups only the other readers make. Output: `reader_writes_probe.txt`.

**The instruments count a known write first (A2)** — `## Run 0`:
- a deliberate miss, `object_get("tolerance", tol=1e-99)`: `shard0000: 1 commit(s); rows tolerance +1`
  on each of the three shards, `primary: 3 commit(s)`, and `_replicated_write: get tolerance x1 (1
  inserted …)`. The primary's three commits are the record's insert, its `store_id` update and its
  delete (read: `_replicated_write` steps 1, 3, 5);
- the same get again, a hit: `primary: 2 commit(s)` and nothing on any shard;
- a deliberate sharded `object_store` of one `GkSourcePolicyData`: `shard0002: 1 commit(s); rows
  GkSourcePolicyData +1`.

So the instrument separates an insert (shards and primary), a hit (primary only) and a sharded write
(one shard).

**On the full store** — `## Run 1`:
- the constructor and `__exit__`: `no file changed`;
- **21** replicated `object_get`s: `get store_tag x1, get LambdaCDM x1, get QCD_Cosmology x1, get
  tolerance x8, get BackgroundModel x2, get wavenumber_exit_time x6, get GkSourcePolicy x2`, every
  one a hit except `QCD_Cosmology`'s `BackgroundModel`, which misses without inserting;
- **`TOTAL commits per file: {'store.sqlite': 42}`** and `TOTAL rows per table …: {}`: two commits
  on the primary per replicated `object_get`, **no row inserted anywhere, and no commit on any
  shard**. The independent open-to-close check agrees: `WHOLE RUN … primary: 42 commit(s)`;
- every `read_table` and every sharded lookup — `object_read_batch QuadSourceIntegral`, `GkSource`,
  `GkSourcePolicyData` (a hit, `available=True`, and a miss), `QuadSource`, `GkNumericIntegration`,
  `GkWKBIntegration`, `TkNumericIntegration`, `TkWKBIntegration` — `no file changed`;
- `in-flight records left on the primary: []`.

**No reader writes a sharded row** (the prompt's fourth stop condition is not met). **Every reader
writes and clears one in-flight record per replicated `object_get`**, and nothing else, on a full
store. For QSI the count is `1 + 2 + (4 + 1 + N_src + N_resp + 2) + (4 + 1)` replicated gets on a
store holding LambdaCDM's background and not QCD's (the formula is [read] from R1; the instance
`N_src = N_resp = 3` is measured as 21).

**On a store missing one row** — `## Run 2` to `## Run 9`, each a copy of the full store with one
row deleted from every shard while closed:

| Row missing | What the reader wrote | What happened |
|---|---|---|
| the run's `store_tag` | 40 primary commits; no row | `resolve_run_selection` found no run, read the store as "unknown generation" with `tags=[]`, and made no `store_tag` lookup |
| `LambdaCDM` | `LambdaCDM +1` on each shard, 1 commit each; 43 primary commits | inserted, serial 1 (the class's only row, so the broker's next serial was the deleted one's) |
| `QCD_Cosmology` | `QCD_Cosmology +1` on each shard; 43 primary commits | inserted, serial 1 |
| the `DEFAULT_QUADRATURE_RTOL` tolerance | `tolerance +1` on each shard; 43 primary commits | inserted under a **new** serial, 9, so rows keyed on the old serial no longer match |
| LambdaCDM's `BackgroundModel` | 16 primary commits; no row | no insert; exit 1 at LambdaCDM (`Could not locate suitable background model…`) |
| one `wavenumber_exit_time` | 32 primary commits; no row | no insert; the sequence raised at the first use of the unpopulated exit time's serial |
| the 5.0 `GkSourcePolicy` | `GkSourcePolicy +1` on each shard; 43 primary commits | inserted under serial 2 |
| the 1.5 `QuadSourcePolicy` (`read_table`) | 32 primary commits; no row | refused (`find_QuadSource_policy: 0 rows`) |

Every insert is a replicated insert on every shard; none is sharded.

### R3. `QCD_Cosmology` in A3 v2, and each reader on a store without a model's background

- **A3 v2's store will hold a `QCD_Cosmology` row** [read]. `run_build`'s `_filtered_build`
  (`docs/handover/quadsource_atol_sweep.py`) calls `_real_build(pool, units)`, which is
  `config.model_list.build_model_list`, and only then keeps LambdaCDM. `build_model_list`
  `object_get`s `QCD_Cosmology` (L15–17), and `sqla_QCDCosmology_factory.build` inserts on a miss.
  So the new store's first open inserts it, as a replicated row, and no `QCD_Cosmology`
  `BackgroundModel` follows, since the filtered list drives the pipeline. `scoped_pipeline_run.py
  --models` filters the same way, after the same real call (read: `main`). Corroborated on the one
  store there is [measured]: the A3 baseline's fingerprint lists `QCD_Cosmology: 1` and
  `BackgroundModel: 1` (`fingerprint_time_probe.txt`, `classes (count):`).
- **Each reader on a store lacking a model's `BackgroundModel`** [read; measured for QSI]. For the
  second model, `QCD_Cosmology`, each script prints `>> RUNNING PIPELINE FOR MODEL QCD_Cosmology`,
  makes its tolerance `object_get`s (2 or 4, hits), then the `BackgroundModel` `object_get` (a miss:
  an unpopulated object, no insert), and raises `RuntimeError("Could not locate suitable background
  model instance in the datastore")` from `if not model.available:` (Gk L329–332, GkWKB L371–374,
  GkSource L780–783, TkWKB L392–395, tensor L323–326, QSI L1074–1077). Nothing catches it, so the
  `with ShardedPool` block exits through `__exit__` and the script exits 1. On the way, each
  replicated get commits and clears a record: for QSI, `[QCD_Cosmology] tolerance x4 … primary: 8
  commit(s)` and `[QCD_Cosmology] BackgroundModel … raised RuntimeError … primary: 2 commit(s)`
  (`reader_writes_probe.txt`, Run 1).
- **For 03's model filter** [inferred, from R1 and M2c]: on a read-only pool `build_model_list`'s
  own `QCD_Cosmology` lookup would refuse on a store that lacks the cosmology row (as a store built
  by neither driver might), before any model is skipped. "A model the store does not hold" has two
  forms — no cosmology row (A3 v2 has one) and no `BackgroundModel` (A3 v2's case) — and 03's
  filter has to meet both.

---

## S. The `SIGTERM` path

Each S probe is `sigterm_probe.py` with one item. Each variant is a child process that starts a
local Ray instance (`ray.init(num_cpus=1, include_dashboard=False)`), submits a task that sleeps,
sends itself `SIGTERM` 2 s later from a timer thread while the main thread is in `ray.wait`, and
reports with times relative to the signal. Before and after each variant the runner lists every
`gcs_server|raylet|ray::` process, records the child's own descendants while it runs, checks which
still exist at +0.5, +2, +5 and +10 s after the child exits, and kills only its own survivors.

### S1. A handler installed after `ray.init` [measured, `sigterm_probe_S1.txt`]

- **What `ray.init` installs.** Before it, `C SIGTERM: SIG_DFL`. After it, the C-level `SIGTERM`
  handler is `_raylet.so+0xd8fb38 = nm __ZN4absl12lts_20230802L24AbslFailureSignalHandlerEiP9__siginfoPv`
  (absl's failure-signal handler, statically linked into Ray's `_raylet.so`), and the Python-level
  one is `init.<locals>.sigterm_handler`, which is `sys.exit(signum)` [read:
  `ray/_private/worker.py` L1498–1502]. A driver handler installed before `ray.init` — as both
  drivers' `terminal` is — is replaced by it [read: `ray.init` calls `set_sigterm_handler`
  unconditionally on the main thread].
- **Ray's handlers kept** (variant `keep-alarm0-sleep`, prompt 10's comparison): the signal
  prints `*** SIGTERM received at time=… ***` and a C++ stack through `ray::core::CoreWorker::Wait()`,
  and the frame around `ray.wait` catches `builtins.SystemExit: code=1 args=(1,)` at `t=+0.010s`.
- **A Python handler installed after `ray.init`** replaces absl's at the C level: `[after
  re-installing a Python handler] C SIGTERM: Python+0x1e24fc (dladdr nearest signal_handler)`. Then,
  in every variant:
  - **it is the handler that runs**, on the main thread, inside the wait, within milliseconds:
    `Python handler ran: signum=15 thread=MainThread frame=wait` at `t=+0.002s`
    (`frame=get_objects` when blocked in `ray.get`);
  - **no banner is printed and no alarm is armed**: `banner printed: 0 time(s)`, and `signal.alarm(0)
    returned 0; nothing to re-arm`; each such child survived 8 s of cleanup.
- **What reaches the frame depends only on what the handler raises:**

  | Handler | Caught around `ray.wait` (or `ray.get`) |
  |---|---|
  | raises `SystemExit(143)` | `builtins.SystemExit: code=1 args=(1,)` at `t=+0.003s` (`ray.get`: the same, `t=+0.001s`) |
  | raises `KeyboardInterrupt(15)` | `builtins.KeyboardInterrupt: code=None args=()` at `t=+0.001s` |
  | raises a `BaseException` subclass | nothing until `ray.wait`'s 30 s timeout; then `builtins.SystemError: … returned a result with an exception set` at `t=+28.116s` |
  | returns | `ray.wait returned normally: 0 ready` at its timeout, `t=+4.017s` |

  So **the code is lost**: whatever `SystemExit` the handler raises arrives as `SystemExit(1)`;
  `KeyboardInterrupt` keeps its type and loses its arguments; anything else is held until the wait
  ends by itself. That the conversion is done by Ray's Cython signal check is **inferred**, from
  `_raylet.so`'s strings (`ray._raylet.check_signals`, `IntentionalSystemExit`); the behaviour is
  measured. `RayWorkPool`'s `ray.wait` has no timeout (read: `RayTools/RayWorkPool.py`, the main
  loop), so a handler that does not raise `SystemExit` or `KeyboardInterrupt` would leave the
  pipeline running.

### S2. Ray's abort [measured, `sigterm_probe_S2.txt` and `sigterm_probe_S3.txt`]

**What it is.** With Ray's handlers kept, the `SIGTERM` reaches absl's `AbslFailureSignalHandler`
first. After it has run:
- `C SIGALRM: _raylet.so+0xd8ff6c = nm __ZN4absl12lts_20230802L27ImmediateAbortSignalHandlerEi`,
  where before the signal it was `SIG_DFL`;
- `signal.alarm(0)` in the cleanup returns **3**, the seconds left on an alarm that was not
  pending before the signal (`alarm pending after ray.init: signal.alarm(0) returned 0`). It
  returned 3 in all 11 such calls across the three files;
- `C SIGTERM: Python+0x1e24fc (dladdr nearest signal_handler)`: absl has handed `SIGTERM` back to
  the handler it displaced, so a second `SIGTERM` goes to Python directly.

Without `alarm(0)` the process dies by **`SIGABRT`** (`exit status: -6 (killed by SIGABRT; shell
status 134)`), in 6 of 6 runs. **The abort is a 3 s `SIGALRM` armed by absl's failure-signal
handler inside Ray's `_raylet.so`, whose `SIGALRM` handler raises `SIGABRT`.** That much is
measured: the alarm's 3 s from `alarm(0)`'s return, both handlers from `sigaction` resolved against
`nm`, the abort from the exit status. The C++ source is not in the installed package and was not
downloaded; `nm` shows `ray::RayLog::InstallFailureSignalHandler(char const*, bool)` and the strings
include `*** %s received at time=%ld%s ***`. That absl's default option sets the 3 s (its
`alarm_on_failure_secs`) is **inferred** from absl's documented default, not read here.

**The window.** The last tick of an 8 s cleanup was at `t=+2.791s`, `+2.806s`, `+2.791s` (S2) and
`+2.811s`, `+2.810s`, `+2.800s` (S3). The runner saw the exits at 3.029, 3.055, 3.033, 3.049, 3.043
and 3.005 s after the signal, under load averages between 7.0 and 9.6. **The process dies between
2.78 and 3.06 s after the signal.**

**Does `alarm(0)` stop it? Yes.** `signal.alarm(0)` from Python, made in the cleanup straight after
the exception, cancelled it every time it was tried: `cleanup: signal.alarm(0) returned 3 (the
alarm is cancelled)` followed by `survived 8 s of cleanup` in S1 (1), S2 (2) and S3 (2 sleeping
variants), and in every S3 variant that went on to exit by itself. The prompt's third stop condition
is not met. A handler installed after `ray.init` (S1) avoids the alarm altogether, since absl's
handler never runs.

### S3. What leaves a `gcs_server` behind [measured, all three files]

| How the driver ended | Runs | Left a `gcs_server` |
|---|---|---|
| `sys.exit` with no explicit `ray.shutdown()` (Ray's `atexit` hook runs it) | S1: 6, S2: 2, S3: 1 | 0 |
| explicit `ray.shutdown()` (1.49 s), then `sys.exit` | S3: 1 | 0 |
| Ray's abort (`SIGABRT` at ~3 s) | S2: 3, S3: 3 | **4** (S2: 2, S3: 2) |
| `os._exit(1)` at once: no `ray.shutdown`, no `atexit` | S3: 3 | **2** |
| `os._exit(1)` 3 s after the signal | S3: 3 | **1** |

**So it is not the abort as such: any exit that skips `ray.shutdown()` can leave the `gcs_server`,
and none always does.** Every exit through `ray.shutdown()`, 10 of 10, left nothing. Each
`gcs_server` left behind had logged `GCS server received SIGTERM, shutting down...` and `Stopping GCS
server.`, then nothing more (`last lines of …/gcs_server.out`); each outlived the runner's own
`SIGTERM` by 3 s and died at `SIGKILL`. Every process the driver started was in the driver's process
group (`pgid=` equal to the child's pid), as `start_reaper`'s `os.setpgrp()` makes it [read:
`ray/_private/services.py` `start_reaper`]. Ray's reaper, on its parent's death, sends `SIGTERM` to
the group and, after a 1 s grace, `SIGKILL` [read: `ray/_private/ray_process_reaper.py`
`reap_process_group`]. Why that `SIGKILL` did not reach these `gcs_server`s is **not established**:
the reaper was gone by +0.5 s in every run, those that left nothing included, which is sooner than
its 1 s grace would allow if it had waited to send it.

### S4. How long the fingerprint takes [measured, `fingerprint_time_probe.txt`]

- **The read.** `RunRegistry.stores.fingerprint_store(primary, write=False)` on
  `var/datastores/handover-A3-baseline-lambdacdm.sqlite`, nothing else: no `Run.finish`, no
  `RunRegistry.begin`, no `store` subcommand, no `ShardedPool`. The five store files and the sidecar
  are `unchanged` by SHA-256, size and mtime, and no journal appeared.
- **The store:** 349,306,880 bytes (333.1 MiB), primary plus four shards.
- **HEAD's fingerprint does not refuse the format-1 store.** It read it and reported problems that
  the store's age explains (`GkSource: absent-table`, `incomplete`, `unresolved-parent`), and its
  comparison with the sidecar is one entry: `the recorded fingerprint is format 1 and this one is
  format 6; fingerprints of different formats are not compared`. `wrote: False`.
- **The time:** **2.480 s** for the first call (load 5.1), 0.683 s for a second, warm call; the
  import was 0.017 s.
- **Against the window.** 2.48 s is inside the 2.78–3.06 s window of S2, by ~0.3 s, **on this
  store**. The A3 baseline holds no `GkSource`, `GkSourcePolicyData` or `QuadSourceIntegral`
  record (`GkSource: 0`, `QuadSourceIntegral: 0`), which A3 v2 will hold in quantity; that A3 v2's
  fingerprint takes longer is **inferred** from that, not measured. The earlier figure, log 05's
  6.66 s for the whole `store fingerprint` command on the live A3 store (interpreter start and
  imports included, 2026-09-25), is already outside the window. Prompt 10's `resumed-b`, whose
  driver ran `pool.__exit__` and then `finish(fingerprint=True)` after the signal, was killed at
  3.0 s and recorded nothing; S2 is the mechanism.

---

## V. The version row

### V1. The order of events at open, and the crash window [read; measured]

**`ShardedPool.__init__`, a new store** [read]: the primary's absence is checked and every shard
file's absence; `_create_engine`; `_write_shard_data` (the primary's six tables, then their rows in one
transaction); the broker; the first actor, the **last** shard id (`shard_ids.pop()`), whose
`Datastore.__init__` creates its file and every table (`_ensure_tables`) and then inserts the version
label's row through `self.object_get("version", label=…)` (serial from the engine; `version` is
exempt from the broker's lease in `_insert`); `ray.get(shard0.object_get("version", …))` for the
serial; every other actor, each with `version_serial`, each creating its file and tables and
inserting the row under that serial; `read_largest_store_ids` and the broker's notification.

**An existing store** [read]: the primary journal is noted; `_read_shard_data`; `_check_shard_files`;
`_reconcile_replicated_tables` (the check at open, which compares the `version` table across shards
like every replicated table); `replication_in_flight.create(checkfirst=True)`; then the broker and
the actors as above, each running `_drop_actions`, `_ensure_tables` and `_validate_on_startup`
before its version lookup, which inserts only under a label the store has not seen.

**Where a crash leaves what** [measured, `version_window_probe.txt`], with every actor after the first
killed:

| Case | On disk after the crash | The next open |
|---|---|---|
| 1. new store, killed before their files exist | `store.sqlite`, `store-shard0002.sqlite` with `[(1, 'L1')]` | refused by `_check_shard_files`: `shard #0 … which does not exist; shard #1 … which does not exist` |
| 2. new store, killed before the version insert | shard 0 created with no version row; shard 2 `[(1, 'L1')]` | refused the same way (shard 1 does not exist) |
| 3. store under `L1` opened under `L2`, killed before the version insert | shard 2 `[(1, 'L1'), (2, 'L2')]`, shards 0 and 1 `[(1, 'L1')]` | **`ReplicatedDivergence`**: `class "version", shard [0, 1], serial 2: held by shard(s) [2], missing on shard(s) [0, 1]` |

The stand-in builds actors one at a time, so in case 2 shard 1's constructor never ran; under Ray
the constructors run concurrently, and which shards hold the row would vary [inferred, from
`ShardedPool.__init__`'s comprehension of `.remote()` calls]. Case 3 is log 02's observation 1, now
committed.

**Two more windows of the same kind.**
- Cases 1 and 2 are **not the version row**: a new store's first open, interrupted before every
  shard file exists, leaves a primary whose next open is refused for a missing shard. The store
  holds nothing, so nothing is lost; the remedy is to delete it, which a person must do, since
  `--build` refuses an existing path and `--build --resume` re-checks the store first. Opened as
  `[00-a-new-stores-first-open-can-leave-a-primary-without-its-shards]`.
- **The prune at open** writes a replicated class the same way. Case 4: a store holding an
  unvalidated `BackgroundModel`, opened with `prune_unvalidated=True` (`main.py`'s default, and
  A3 v2's resume once prompt 05 lands), every actor after the first killed: shard 2's
  `BackgroundModel (serial, validated) []`, shards 0 and 1 `[(1, 0)]`, and the next open is refused
  with five differences (`BackgroundModel`, three `BackgroundModelValue`, one `BackgroundModel_tags`).
  Each actor prunes its own shard in `_validate_on_startup`, in its constructor, outside the record
  [read: `sqla_BackgroundModelFactory.validate_on_startup`]. Opened as
  `[00-the-prune-at-open-deletes-a-replicated-row-outside-the-in-flight-record]`.

### V2. The forms 02 can take, and how a read-only open finds the row [read]

**What each actor needs the serial for.** `Datastore._insert` writes `self._version.store_id` into
every row of a class registered with `"version": True` (13 factories); nothing else reads it, and no
factory's query filters on a `version` column (read: `Datastore._insert`; a search of
`Datastore/SQL/ObjectFactories/` finds no `c.version`). So an actor needs the serial before its
first insert of such a class, and never for a read.

**Forms that work**, each with what it changes in `Datastore.__init__`:
1. **A — no version work in the constructor; the pool writes the row and tells the actors.**
   `Datastore.__init__` drops `version_label`/`version_serial` and the `object_get("version", …)`;
   the actor gains `set_version(serial)`, and `_insert` refuses a `"version": True` insert before
   it is called. After every actor exists, `ShardedPool.__init__` does `self._version =
   ray.get(self._get_impl_replicated_table("version", {"label": label}))` — through
   `_replicated_write`, so a crash leaves a `get`/`version` record whose controlling shard holds the
   row, which `_plan_replicated_repair` already repairs for any class by copying the controller's
   rows (read: its generic `get` branch) — and then `set_version` on every actor.
2. **B — the constructor looks the row up without inserting it.** As A, but `Datastore.__init__`
   keeps the label and selects the row by label, setting the serial when it is found; the pool
   writes it through `_replicated_write` only if some actor lacks it, then calls `set_version`.
   It works, at the cost of a second path to the same state.
3. Not a form: keeping the constructor's insert and widening the check at open's licence for a
   `version` row missing under the label being opened. 02's charter excludes it.

**Recommendation: A.** One path writes the row, it is the path every other replicated row takes,
and the check at open needs no special case. In a new store, the row is then written after every
shard file and table exists, so a crash before it leaves consistent shards with no version row,
which the next open writes; a crash during it leaves a record the check at open repairs.

**How a read-only open finds the row.** By label, with a select on a file opened `mode=ro`:
`SELECT serial, label FROM version WHERE label = ?` returned `[(1, 'standin')]` on each shard, and
`[]` for a label the store does not hold [measured, `readonly_open_probe.txt` `## M2e`]. Reads do
not use the serial, so for a read-only pool the row is an identity check only. When it is absent:
**refuse**, naming the class, the label asked for and the labels present — that follows from 03's
charter, "a lookup that misses refuses instead of inserting". When some shards hold it and others
do not, the check at open, kept read-only, already refuses (V1 case 3).

---

## M. What a read-only pool needs

### M1. Every write at open, and what `read_only=True` does with each

What a read-write open writes today [measured, `readonly_open_probe.txt` `## M1`]:
- a clean full store, `prune_unvalidated=False` (the readers'): **no file changed**;
- an unvalidated `BackgroundModel`, `prune_unvalidated=False`: no file changed (so
  `_validate_on_startup` only reads when not pruning);
- the same, `prune_unvalidated=True`: one commit per shard, `BackgroundModel -1,
  BackgroundModelValue -12, BackgroundModel_tags -1`;
- `drop_actions=['gk-source-policy']`: six commits per shard, `GkSourcePolicyData -1` on the one
  shard that held one (the table dropped, then created again by `_ensure_tables`);
- a primary without `replication_in_flight`: `primary: 1 commit(s); new tables replication_in_flight`;
- shard 1 lacking the `OneLoopIntegral` tables: ten commits on shard 1, `new tables OneLoopIntegral,
  OneLoopIntegral_tags`;
- a version label the store has not seen: `version +1` on every shard;
- a path where no store exists: every file `appeared` — **a reader given a wrong `--database`
  creates a store there** (read: `ShardedPool.__init__`'s new-store branch; the readers pass the
  path unchecked).

| Write at open (where) | Under `read_only=True` | Reason |
|---|---|---|
| New-store branch: `mkdir`, `_write_shard_data`, every shard file and its DDL (`ShardedPool.__init__`, `Datastore.__init__`) | **refuse** | A reader never creates a store; today a wrong path silently makes one (M1 above). |
| Hot-journal rollback of the primary: its first read-write connection, in `_read_shard_data` | **refuse** if a journal is beside the primary | `store_reader`'s rule. A `mode=ro` connection cannot roll it back: it raises `SQLITE_READONLY_ROLLBACK` (M2d). |
| Hot-journal rollback of each shard: `_reconcile_replicated_tables` step 1, a read-write `sqlite3.connect` | **refuse** if a journal is beside any shard | Measured as a write today, even with every engine `mode=ro`: the M2b pool `opened … hot_journals=[…shard0001…]` and the journal `went` (M2d). |
| The comparison of replicated tables (`_reconcile_replicated_tables` step 2) | **keep**, on `mode=ro` connections | It only reads, and it is what refuses a diverged store, as V1 case 3 shows. |
| Repair of an interrupted replication, and clearing the record (steps 4–5) | **refuse** when the primary holds a record | A repair is a write; a store with a record is mid-write. |
| `replication_in_flight.create(checkfirst=True)` | **skip** | DDL. `_read_in_flight_record` already reads a missing table as no record. |
| `_drop_actions` (`Datastore.__init__`) | **refuse** if any is passed (`ReadOnlyWrite`) | DDL. The readers pass none. |
| `_ensure_tables` | **skip**; report the tables a shard lacks | DDL. A lookup on a missing table then fails in SQLite with `no such table`; `store_reader` reports absent tables the same way and does not refuse. |
| `_validate_on_startup` with `prune_unvalidated=False` | **keep** | It only reads and prints (measured above). |
| The prune, `prune_unvalidated=True` | **refuse** (`ReadOnlyWrite`) at construction | It deletes rows (measured above). |
| The version insert in `Datastore.__init__` | **skip**: a lookup by label; **refuse** when absent | V2. |
| The broker, `read_largest_store_ids`, `notify_largest_store_ids` | **skip** | Nothing on a read-only pool allocates a serial. They write no file; skipping them is economy, not safety. |
| `SerialPoolManager` in each actor | **keep** (inert) | It contacts the broker only when an insert leases a serial (read: `SerialPoolManager.lease_serial`). |
| `ShardedPool.__exit__`, `Datastore.__exit__` | **keep** | They dispose engines and return leases; no file write (`ShardedPool close … no file changed`, R2). |

No row is left "to be decided".

Per lookup, under `read_only=True` [read, with M2 measuring the effect]:
- **a replicated `object_get`** must not go through `_replicated_write`: one shard's actor answers,
  as `read_table` does, with no record and no replication;
- **a miss** must refuse before it inserts. Every insert-on-miss in the replicated factories goes
  through the `inserter` the actor passes to `build` (read: R1's factories), so an inserter that
  raises `ReadOnlyMiss(class, payload)` covers every miss without a per-factory change;
- **anything else that writes** — `wavenumber`'s and `redshift`'s flag updates on a hit, `validate`,
  `store` — is stopped by the `mode=ro` file itself, which raises `sqlite3.OperationalError:
  attempt to write a readonly database` (M2c); 03 can translate that into `ReadOnlyWrite`.

### M2. Every reader lookup served `mode=ro` [measured, `readonly_open_probe.txt`]

Every engine was opened with `store_reader.read_only_url` by patching `ShardedPool._create_engine`
and `Datastore._create_engine`, and nothing else.

- **M2a, today's pool, read-only.** `the constructor returned` and `files after the open: no file
  changed`, on a clean full store. The first replicated `object_get` — a hit — raised
  `sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) attempt to write a readonly database`:
  the in-flight record's insert on the primary. `read_table wavenumber returned`.
- **M2b, the lookups an actor can serve.** With each replicated `object_get` sent to one shard's
  actor, with no record and no replication, QSI's whole sequence and the four other sharded lookups
  ran on the full store exactly as in R2 Run 1 — the same hits (`available=True, serial=1`, `6 of 6
  available`, `12 source, 6 response`, `serials 1, 6`, `GkSourcePolicyData … available=True`), the
  same `RuntimeError` at `QCD_Cosmology`'s background — with `TOTAL commits per file: {}` and `M2b+M2c,
  every file, open to close: no file changed`. **Every lookup the six readers make can be served
  from `mode=ro` files**, provided the replicated ones bypass `_replicated_write`.
- **M2c, an attempted insert raises from SQLite.** On the same pool, `object_get tolerance
  tol=1e-99` and `object_get GkSourcePolicy (9.0)` each raised `sqlalchemy.exc.OperationalError:
  (sqlite3.OperationalError) attempt to write a readonly database`, from the `INSERT` (printed by
  `Datastore.object_get`'s handler with its SQL). The serial in the failed `INSERT` came from a
  broker lease, so the insert path ran to SQLite before failing; a refusing inserter would stop it
  earlier.
- **M2d, journals.**
  - an empty `-journal` beside shard 1: `store_reader.open_read_only … refused` (`… so the store is
    open or was not closed cleanly. Nothing was read or repaired`). **The read-only pool did not**:
    `opened and served a lookup`. SQLite does not treat an empty journal as hot, so 03 must make the
    refusal itself, as `store_reader` does, by the presence of the file;
  - a **hot** journal, made by a child that died inside a write transaction on shard 1 after its
    cache spilled (`journal exists=True, size=10752`): a `mode=ro` read raised `OperationalError:
    attempt to write a readonly database (sqlite_errorcode=776, sqlite_errorname=SQLITE_READONLY_ROLLBACK)`,
    through SQLAlchemy and through `sqlite3` alike; `store_reader.open_read_only … refused`; no file
    changed. The read-only-engine pool and today's read-write pool each rolled it back
    (`hot_journals=[…shard0001.sqlite]`, `store-shard0001.sqlite: changed, counter 191 -> 191;
    store-shard0001.sqlite-journal: went`). A rollback restores page 1, change counter included, so
    the counter does not see it; the SHA-256 does.
- **The readers' own run-label read** already goes through `store_reader` (`read_inventory`), so it
  refuses a journal today (read: `extract_common.available_run_labels`).

---

## Issues

Opened by this audit, on [`prompts/a3-v2-readiness/IMPLEMENTATION_STATE.md`](../prompts/a3-v2-readiness/IMPLEMENTATION_STATE.md) §3.2:
- `[00-the-prune-at-open-deletes-a-replicated-row-outside-the-in-flight-record]` — V1 case 4. Does
  not block A3 v2: only an interruption of a pruning open of a store that holds an unvalidated
  `BackgroundModel` reaches it.
- `[00-a-new-stores-first-open-can-leave-a-primary-without-its-shards]` — V1 cases 1–2. Does not
  block A3 v2: the store is empty and can be deleted.

Narrowed or widened, on the `datastore-integrity` board:
- `[08-extraction-script-object-get-of-gksourcepolicy-writes-to-the-store-it-reads]` — **widened**:
  every replicated `object_get` every reader makes writes and clears a record (21 for QSI on R2's
  store); five classes insert on a miss (R1), four of them measured (R2: the run's `store_tag`
  lookup is not reached on a miss); no reader writes a sharded row.
- `[10-a-sigterm-to-the-scoped-driver-is-recorded-as-failed-or-not-at-all]` — **widened**: both
  inferred mechanisms are now measured as behaviour; the abort's identity is measured; the
  `gcs_server` is left by any exit without `ray.shutdown()`, not by the abort as such.

---

## What 02, 03 and 04 can now be written against

### 02 — the version row

**Facts its method rests on.**
- The window is the first actor's version insert to the last actor's, in actors' constructors,
  outside the record; a crash in it under a new label leaves a store the check at open refuses (V1
  case 3, measured).
- Actors need the serial only to insert `"version": True` rows; reads never use it (V2, read).
- `_replicated_write` and `_plan_replicated_repair` already handle a `get` of any replicated class,
  and the check at open already compares the `version` table across shards (V2, read).
- A read-only open can find the row by label `mode=ro` without inserting it (M2e, measured), and
  should refuse when it is absent (V2, from 03's charter).

**Method recommended:** form A of V2 — actors built without the version row, the pool writing it
once through `_replicated_write` after every actor exists, then `set_version` on each actor, and
`_insert` refusing a versioned insert before it.

**Decisions for the user.**
- **U1 — does 02 also put the prune of a replicated row through a record?**
  `[00-the-prune-at-open-deletes-a-replicated-row-outside-the-in-flight-record]` is the same defect
  as the version row's, for the other replicated write a constructor makes, and prompt 05 makes
  A3 v2's resume prune at open. The repair cannot simply be reused: it never deletes a replicated
  row (read: the comment above `_reconcile_replicated_tables`). **Recommendation: yes, in 02.** The
  pool, not the actors, prunes replicated classes, before any actor exists and with no Ray, under
  a record naming a `prune` of the class; the prune is idempotent (it deletes the unvalidated rows,
  and only those, from every shard), so the check at open repairs a `prune` record by running it
  again. Sharded classes keep their per-shard prune, which needs no record.
- **U2 — what is done about `[00-a-new-stores-first-open-can-leave-a-primary-without-its-shards]`?**
  **Recommendation: record it, change no code.** Nothing is lost, and form A already moves the
  version row after the shard files exist; 07's rehearsal can confirm the remedy (delete the empty
  store, launch again) is clear from the refusal.

### 03 — open readers read-only

**Facts its method rests on.**
- R1 is the complete list of reader calls. Replicated `object_get`s: `store_tag`, `LambdaCDM`,
  `QCD_Cosmology`, `tolerance`, `BackgroundModel`, `wavenumber_exit_time`, `GkSourcePolicy`.
  `read_table`s: `wavenumber`, `redshift`, `QuadSourcePolicy`. Sharded: eight lookups, none of which
  writes (R1 read, R2 measured).
- On a full store the only write is the in-flight record, two primary commits per replicated
  `object_get` (R2 Run 1, measured); `store_tag`, `LambdaCDM`, `QCD_Cosmology`, `tolerance` and
  `GkSourcePolicy` insert on a miss, on every shard (R2 Runs 2–9, measured).
- `mode=ro` files serve every reader lookup if replicated lookups skip `_replicated_write` (M2b);
  today's pool cannot, because the record's insert fails (M2a); a miss raises from SQLite (M2c).
- M1's table: what `read_only=True` skips, refuses or keeps at open, row by row, with no row left
  to decide.
- An empty journal is not refused by SQLite and must be refused by presence, as `store_reader` does;
  a hot journal makes a `mode=ro` read fail with `SQLITE_READONLY_ROLLBACK`; the check at open's
  read-write `sqlite3.connect` rolls a hot journal back, so it is a write 03 replaces (M2d).
- A3 v2's store holds a `QCD_Cosmology` row and no `QCD_Cosmology` background, and every reader
  exits 1 at that model's background lookup (R3); on a read-only pool, `build_model_list` itself
  would refuse on a store without the cosmology row (R3, inferred).

**Method recommended.**
- A replicated `object_get` on a read-only pool goes to one shard's actor, like `read_table`, with
  no record.
- The actor passes `build` an inserter that raises `ReadOnlyMiss(class, payload)`, which covers
  every insert-on-miss in one place.
- The `mode=ro` engine is the backstop for any other write, translated into `ReadOnlyWrite`.
- `--models` is applied before any cosmology lookup, and a selected model with no `BackgroundModel`
  is skipped with a message.

**Decisions for the user:** none found.

### 04 — record a `SIGTERM` as killed

**Facts its method rests on.**
- A Python handler installed after `ray.init` replaces absl's at the C level. It runs on the main
  thread inside `ray.wait` and `ray.get` within milliseconds. No banner is printed and no alarm is
  armed, so nothing aborts the process (S1, measured).
- What reaches the frame: a `SystemExit` raised by the handler arrives as `SystemExit(1)`;
  `KeyboardInterrupt` keeps its type; anything else is held until the wait ends, and `RayWorkPool`'s
  wait has no timeout (S1, measured; RayWorkPool read). **So the handler must record the signal
  itself** — write `killed`/143 at once — and then raise `SystemExit` or `KeyboardInterrupt`. Later,
  `terminal_state(1)` must not overwrite that record with `failed`.
- If Ray's handler runs instead, the process dies by `SIGABRT` 2.78–3.06 s after the signal unless
  `signal.alarm(0)` is called first, which stops it (S2, measured).
- Exits without `ray.shutdown()` left a `gcs_server` in 7 of 12 runs; exits through it, in none of
  10. An explicit `ray.shutdown()` took 1.49 s (S3, measured).
- The fingerprint took 2.48 s on the A3 baseline (333 MiB, partial), and will take longer on A3 v2
  (S4, measured; A3 v2 inferred). It cannot be relied on inside 3 s, so it comes after the state is
  written, with no alarm armed.
- Today's drivers install `terminal` **before** `ray.init`, which replaces it, so `terminal` never
  runs once Ray is up (S1, read). `ray.init` is reached through each driver's `_local_ray_init`
  wrapper, which is where the handler can be installed after it (read: `run_build`,
  `scoped_pipeline_run.main`).

**Method recommended.**
- One function, called in each driver's `_local_ray_init` after the real `ray.init` returns,
  installs a handler for `SIGTERM` and `SIGINT`.
- On its first call the handler writes `killed`, `exit_code=128+signum`, and remembers that it did.
  It then raises `SystemExit(128+signum)`, knowing that it will arrive as `SystemExit(1)`.
- The drivers' `except SystemExit` path sees the remembered signal and keeps `killed`. It calls
  `signal.alarm(0)` (a guard for a signal that reached Ray's handler before the driver's was
  installed), takes the fingerprint best-effort, and calls `ray.shutdown()`.
- A second signal during this cleanup is ignored by the handler.

**Decisions for the user:** none found that the charter does not settle. One consequence to note:
with the handler installed, a `SIGTERM` no longer prints Ray's `*** SIGTERM received` banner and
stack, which is how prompt 10 identified `resumed-a`'s interruption; the handler should print its
own line instead.
