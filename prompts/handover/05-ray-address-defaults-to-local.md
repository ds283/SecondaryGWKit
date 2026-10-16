# Prompt 05 — `main.py` and the six readers start their own Ray instance by default

**Campaign:** [`README.md`](README.md) · **Board item:** **05** (outside workstreams A–E; see §1) ·
**Board:** `IMPLEMENTATION_STATE.md` — exists; add the 05 row to §1 and §2.
**Closes:** `[00-main-and-the-readers-default-to-ray-address-auto]`, which is **held by the
`runregistry-extraction` board** (§3) and was **assigned to this campaign by the user on
2026-10-12**, to land before the A3 v2 launch or with it.
**Recommended model:** **Opus**. The change is seven defaults; the care is in what must *not* move.

**Read first:**

1. [`prompts/runregistry-extraction/IMPLEMENTATION_STATE.md`](../runregistry-extraction/IMPLEMENTATION_STATE.md)
   §3, the entry `[00-main-and-the-readers-default-to-ray-address-auto]` in full — the defect, the
   impact and the two remedies it weighs. Then that campaign's
   [`README.md`](../runregistry-extraction/README.md) §4.1, "Ray addresses".
2. `main.py` — `DEFAULT_RAY_ADDRESS` (`:99`), the `--ray-address` argument (`:251-256`), and the
   module-level `ray.init` (`:307-308`) with the comment above it.
3. The six readers — `extract_Gk_data.py`, `extract_GkWKB_data.py`, `extract_GkSource_data.py`,
   `extract_TkWKB_data.py`, `extract_tensor_source_data.py`,
   `extract_QuadSourceIntegral_data.py` — each one's `--ray-address` argument and its `ray.init`.
4. Ray's own resolution of the address, in the venv's Ray (2.43.0): `ray.init`'s docstring and its
   `RAY_ADDRESS` override (`ray/_private/worker.py`), and
   `ray/_private/services.py`'s `canonicalize_bootstrap_address` and
   `get_ray_address_from_environment`. What `"auto"`, `None` and `"local"` each do is there and
   nowhere else.
5. `docs/handover/quadsource_atol_sweep.py` — `_main_py_argv` and `run_build`'s
   `_local_ray_init`. **Read only.** Then `RunDrivers/tests/test_prune_on_resume.py`, which pins
   `_main_py_argv`'s list element by element.
6. `ComputeTargets/tests/test_reader_read_only.py` and
   `ComputeTargets/tests/test_main_plumbing.py` — the `ast` pattern for scripts that cannot be
   imported.

---

## 1. Why this exists, and why it is here

`main.py` and all six readers default `--ray-address` to `"auto"`. In Ray 2.43.0 that means:
`RAY_ADDRESS` if it is set; otherwise the address in `/tmp/ray/ray_current_cluster`; otherwise
**any `gcs_server` it finds by scanning running processes**, without a warning; and a
`ConnectionError` only if there is none at all. So a build or a reader left at its default
attaches to whatever Ray instance is up on the machine — another project's isolated cluster
(refused today only because the Ray versions differ, which is incidental) or SGK's own build in
flight, where the versions match and the reader's work would be submitted into it.

Nothing has gone wrong yet, because every driver that matters names its address: the A3 v2 build,
`quadsource_atol_sweep.py --build`, replaces `ray.init` with `address="local"`, and the rehearsal's
`run_readers.sh` passes `--ray-address local`. The protection is in the callers. This prompt puts
it in the default, so that a hand-run reader or a bare `main.py` cannot attach by accident.

It sits in this campaign because the user assigned it here on 2026-10-12: `handover` owns the A3 v2
launch, the next long build that runs `main.py`. It is not part of the seam work and belongs to no
workstream, so its board item is its prompt number. It has no prerequisite and nothing depends on
it except that it should land before or with the launch.

---

## 2. What to change

**The remedy is the default, `"local"`.** `ray.init(address="local")` starts a fresh local instance
for this process even when another is running, and Ray ignores `RAY_ADDRESS` for it (the override
applies only to `None` and `"auto"`). Attaching remains possible by asking for it:
`--ray-address auto`, or a concrete `host:port`.

1. `main.py:99`: `DEFAULT_RAY_ADDRESS = "auto"` → `"local"`. The argument keeps
   `default=DEFAULT_RAY_ADDRESS`.
2. Each of the six readers: `default="auto"` → `default="local"`.
3. In all seven, the argument's `help` and the comment above `ray.init` — which says *"defaults to
   'auto' meaning a locally running cluster"* — say what the default now does and how to attach
   instead. Seven comments that contradict the code are worse than none.

**Do not make the flag required.** The issue records this as the other remedy, and it is the one
StochasticInstantons took. Here it breaks the A3 v2 launcher: `_main_py_argv` passes no
`--ray-address`, `run_build` then replaces `ray.init` wholesale, and
`RunDrivers/tests/test_prune_on_resume.py` pins that argv element by element. The launcher is the
rehearsed artefact — **`docs/handover/quadsource_atol_sweep.py` is not edited**, and its SHA-256
must read `2a98530369f530cfc6f33ae91ff78fb51d3cbc89dea56f8bcfb9cb9ee012676d` before and after.

The launcher is unaffected by the new default whichever way it is read: `run_build`'s
`_local_ray_init` ignores its arguments and passes `address="local"` itself, so `main.py`'s
`ray.init(address=args.ray_address)` reaches Ray as `"local"` both before and after this prompt.

---

## 3. The test

One new module, `ComputeTargets/tests/test_ray_address_default.py`, beside the two it borrows its
pattern from. No Ray, no datastore; it must leave Ray uninitialised and say so in a
`tearDownModule`, as `test_reader_read_only` does.

It reads the seven scripts with `ast` (`main.py` cannot be imported, and the readers call `ray.init`
at module scope) and asserts, for each:

- the `--ray-address` argument's default is the string `"local"` — for `main.py`, through the
  module-level assignment `DEFAULT_RAY_ADDRESS = "local"` that the argument names;
- the argument is not `required`;
- there is exactly one `ray.init` call, and its `address` is `args.ray_address` — so the default is
  what actually reaches Ray.

And one test that pins **Ray's** half of the contract without starting Ray: with `RAY_ADDRESS` set
in the environment, `canonicalize_bootstrap_address("local")` returns `None` (start a new
instance), where `"auto"` returns the environment's address. That is the property this prompt
relies on, read from Ray's source; a Ray upgrade that changed it should fail here and not in a
build.

**The breakage record.** Revert each default in turn to `"auto"` (main.py and one reader suffice),
and once make one reader's `ray.init` ignore the flag; run the module; record which test failed;
restore. Record it in the log.

---

## 4. What this prompt does not do

- It does not edit `docs/handover/quadsource_atol_sweep.py` (see §2), `RunDrivers/tests/`,
  `run_readers.sh`, `CLAUDE.md` or `prompts/runregistry-extraction/README.md`.
- It does not touch `run_child`'s `ray.init`, which passes no address at all. That is
  `[04-the-sweeps-run-child-shares-the-sigterm-and-cluster-address-defects]` on the
  `a3-v2-readiness` board, and it is not A3 v2's path.
- It does not move the shared argument into `extract_common`. Seven literal defaults, each pinned
  by a test, are enough; a helper is a refactor this issue did not ask for.
- It launches nothing long, starts no Ray instance, and takes no rehearsal lock.

---

## 5. Acceptance

1. `main.py`'s `DEFAULT_RAY_ADDRESS` and the six readers' defaults read `"local"`; the seven help
   strings and comments say so.
2. The new module passes, and each deliberate breakage of §3 was caught by a named test.
3. All seven unittest roots pass. Recorded counts before this prompt: `AdaptiveLevin` **32**,
   `CosmologyModels` **39**, `ObjectFactories` **279**, `tools` **91**, `LiouvilleGreen` **148**
   (`skipped=1`), `RunDrivers` **99**, `ComputeTargets` **693**. After: the same, except
   `ComputeTargets` rises by exactly the tests added.
4. `quadsource_atol_sweep.py`'s SHA-256 unchanged.
5. `black --check` clean on every touched `.py` file.
6. One commit; the log, the board and `docs/OPEN_ISSUES.md` in it.

---

## 6. Stop conditions — stop and ask the user

- Any change to `quadsource_atol_sweep.py` or `test_prune_on_resume.py` appears necessary.
- Ray's resolution of `"local"` in the venv is not what §2 says — it honours `RAY_ADDRESS`, or
  attaches to an existing instance.
- A test anywhere in the tree depends on the old `"auto"` default.

---

## 7. The log and the board

`logs/05-ray-address-defaults-to-local.md`, template as the campaign README §5.1, including the
breakage record and the launcher's SHA-256 before and after.

**Board.** The 05 row in §1 and the item table; a §4 record that the issue closed. The issue itself
moves from §3 to §4 **on the `runregistry-extraction` board**, which holds it, with its resolution —
campaign README §5 rule 4. `docs/OPEN_ISSUES.md`: delete its row from §1.1, and correct the count
and the date.
