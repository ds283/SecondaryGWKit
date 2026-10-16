# Log 05 — `main.py` and the six readers start their own Ray instance by default

**Prompt:** [`prompts/handover/05-ray-address-defaults-to-local.md`](../05-ray-address-defaults-to-local.md)
**Commit:** *(this commit)* — "Start a fresh Ray instance by default in main.py and the readers"
**Model:** Claude Opus 5.5
**Date:** 2026-10-16
**Result:** COMPLETE

Written and landed in one session, on `handover-remedial` at `b1e5b7c` (the commit that wrote the
prompt). No orchestrator dispatched it; the prompt's acceptance was checked by the same agent.

## What shipped

**`main.py`**
- `:99` `DEFAULT_RAY_ADDRESS = "auto"` → `"local"`. The argument at `:251-260` keeps
  `default=DEFAULT_RAY_ADDRESS` and is not required.
- `:255-259` help `"specify address of Ray cluster"` → *"address of the Ray cluster to connect to. The
  default, 'local', starts a fresh Ray instance for this run and never attaches to one already
  running; pass 'auto' or host:port to attach"*.
- `:311-313` the comment above `ray.init` (`:314`), which read *"defaults to 'auto' meaning a locally running
  cluster"*, now says what `local` does and how to attach. `ray.init(address=args.ray_address)` is
  unchanged.

**The six readers** — `extract_Gk_data.py`, `extract_GkWKB_data.py`, `extract_GkSource_data.py`,
`extract_TkWKB_data.py`, `extract_tensor_source_data.py`, `extract_QuadSourceIntegral_data.py`:
in each, `--ray-address`'s `default="auto"` → `default="local"`, the same help text as `main.py`,
and the same comment above the one `ray.init(address=args.ray_address)`. `black` split each
argument across lines, which is most of the diff's line count.

**`ComputeTargets/tests/test_ray_address_default.py`** (new, 7 tests, no Ray, no datastore;
`tearDownModule` asserts Ray was not initialised):

- `TestTheDefaultIsLocal` reads `main.py` and the six readers with `ast`:
  `test_main_py_constant_is_local` (the module-level `DEFAULT_RAY_ADDRESS` is the constant
  `"local"`); `test_main_py_argument_defaults_to_the_constant` (the argument's `default` is the
  name `DEFAULT_RAY_ADDRESS`, so the two cannot come apart); `test_each_reader_defaults_to_local`;
  `test_no_script_requires_the_flag`; `test_each_ray_init_takes_the_flag` (exactly one `ray.init`
  per script, no positional argument, `address` is `args.ray_address`).
- `TestRayResolvesLocalToANewInstance` pins Ray's half without starting it:
  `ray._private.services.canonicalize_bootstrap_address("local")` returns `None` — start a new
  instance — with `RAY_ADDRESS` set to `10.0.0.1:6379`, and `"auto"` returns that address.

No other file outside `prompts/` and `docs/OPEN_ISSUES.md`.

## Deviations from the prompt

None. Two choices the prompt left open:

### 1. The help text and comment wording — IMPLEMENTATION CHOICE

The prompt asked that they say what the default does and how to attach. The same sentence is used
in all seven so that a reader of one knows the others. The comment cites "handover prompt 05" so
that the reason for `local` is findable from the code.

### 2. Ray's half is tested through `canonicalize_bootstrap_address` — IMPLEMENTATION CHOICE

`ray.init` resolves `RAY_ADDRESS` in two places in 2.43.0: its own override in
`ray/_private/worker.py` (applied only when the address is `None` or `"auto"`), then
`services.canonicalize_bootstrap_address` (which returns `None`, "start a local instance", for
`"local"` before it consults anything). Only the second can be called without starting Ray. The
alternative, mocking Ray's node start-up and calling `ray.init`, would test a mock. The first is
read, not tested; it is quoted in "Verification performed". The test uses a private module; if a
Ray upgrade moves it, the import fails loudly in this module, which is the intent.

## Verification performed

**I ran these.**

- **Ray's resolution, read in the venv's Ray 2.43.0** (Python 3.12.15).
  `worker.init`'s docstring: *"If the provided address is 'local', start a new local Ray instance,
  even if there is already an existing local Ray instance."* Its override: `if address_env_var and
  (address is None or address == "auto"): address = address_env_var`. And
  `canonicalize_bootstrap_address`: `if addr is None or addr == "auto": addr =
  get_ray_address_from_environment(...)`; `if addr is None or addr == "local": return None`.
  `get_ray_address_from_environment` is where `auto` falls back to *any* running `gcs_server`
  (`len(gcs_addrs) > 0 and addr == "auto"` → `bootstrap_addr = list(gcs_addrs).pop()`), as the
  issue says. So prompt §6's second stop condition does not hold.
- **The new module on the unfixed tree** (`b1e5b7c`, before the code change): 7 tests, `FAILED
  (failures=7)` — `test_main_py_constant_is_local` once and `test_each_reader_defaults_to_local`
  once per reader (six subtests), each `AssertionError: 'auto' != 'local'`. The other five tests
  passed there, as they should: the arguments were already optional and each `ray.init` already
  took the flag.
- **The new module on the fixed tree:** `Ran 7 tests … OK`.
- **The breakage record.** Each made on the fixed tree, the module run, the file restored from a
  copy and `cmp`-checked identical:

  | # | Breakage | Failed |
  |---|---|---|
  | 1 | `main.py`: `DEFAULT_RAY_ADDRESS = "auto"` | `test_main_py_constant_is_local` only |
  | 2 | `main.py`: the argument's `default="local"` as a literal, bypassing the constant | `test_main_py_argument_defaults_to_the_constant` only |
  | 3 | `extract_GkWKB_data.py`: `default="auto"` | `test_each_reader_defaults_to_local` (`reader='extract_GkWKB_data.py'`) only |
  | 4 | `extract_TkWKB_data.py`: `ray.init(address="auto")` | `test_each_ray_init_takes_the_flag` (`script='extract_TkWKB_data.py'`) only |
  | 5 | `main.py`: `required=True` on `--ray-address` | `test_no_script_requires_the_flag` (`script='main.py'`) only |

- **`main.py --help`** prints the new help text and exits at `parse_args`, before `ray.init`.
- **The launcher.** `shasum -a 256 docs/handover/quadsource_atol_sweep.py` →
  `2a98530369f530cfc6f33ae91ff78fb51d3cbc89dea56f8bcfb9cb9ee012676d` before and after; the file is
  not in the diff, nor is anything under `RunDrivers/`.
- **`black --check`** on `main.py`, the six readers and the new test: *9 files would be left
  unchanged.*
- **The seven roots**, before at `b1e5b7c` and after on the fixed tree:

  | Root | Before | After |
  |---|---|---|
  | `AdaptiveLevin` | 32 OK | 32 OK |
  | `CosmologyModels` | 39 OK | 39 OK |
  | `ObjectFactories` | 279 OK | 279 OK |
  | `tools` | 91 OK | 91 OK |
  | `LiouvilleGreen` | 148 OK (skipped=1) | 148 OK (skipped=1) |
  | `RunDrivers` | 99 OK (skipped=1) | 99 OK (skipped=1) |
  | `ComputeTargets` | 693 OK | **700** OK |

  `ComputeTargets` rose by 7, the tests added. `RunDrivers` carries `skipped=1` on both sides; the
  prompt's list of recorded counts did not say so, and it is not this prompt's.

**Reasoned, not run.** The A3 v2 launch is unaffected: `run_build`'s `_local_ray_init` takes
`*_a, **_kw`, discards them and calls the real `ray.init(address="local", …)`, so whatever
`main.py` passes never reaches Ray. That was true before this commit and is true after.

**Not run, by the prompt's rule.** No Ray instance was started, nothing long was launched, and the
rehearsal lock was not taken.

## Observations not acted on

- **The sweep's `run_child` still calls `ray.init` with no `address`.** In Ray 2.43.0 that reads
  `RAY_ADDRESS` and then `/tmp/ray/ray_current_cluster`, so it can attach to a cluster started by
  `ray start`. It is already open as
  `[04-the-sweeps-run-child-shares-the-sigterm-and-cluster-address-defects]` on the
  `a3-v2-readiness` board, it is not A3 v2's path, and the launcher is frozen. No new issue.
- **The readers' `--ray-address` is seven copies of one argument.** A helper in `extract_common`
  beside `add_models_argument` would make one place to change. Not opened as an issue: the test
  pins all seven, so a drift cannot go unseen, and it is a refactor rather than a defect.

## State handed to the next prompt

- `main.py` and the six readers start a fresh local Ray instance unless given `--ray-address auto`
  or a `host:port`; `RAY_ADDRESS` in the environment is ignored at the default. A bare `main.py`
  no longer needs a cluster started first — at `auto` it raised `ConnectionError` when none was
  running.
- Anyone who means to attach a reader to a running build must now say so with `--ray-address`.
- `[00-main-and-the-readers-default-to-ray-address-auto]` is closed on the `runregistry-extraction`
  board's §4, which now has no open issue, and its row is gone from `docs/OPEN_ISSUES.md` §1.1.
