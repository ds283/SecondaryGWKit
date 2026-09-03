# Log 05 — Split `store_handler` into `store_handler` + `persist_handler` (E1)

**Prompt:** prompts/backport-modules/05-persist-handler-split.md
**Commit:** *(SHA intentionally not embedded — see `IMPLEMENTATION_STATE.md` §5 note 11)*
**Date:** 2026-09-04
**Result:** COMPLETE

## What shipped

### Task 1 — `RayTools/RayWorkPool.py`

- Module-level defaults (was `RayWorkPool.py:64-65`): the existing `_default_store_handler(obj,
  pool) -> ObjectRef: return pool.object_store(obj)` is renamed to `_default_persist_handler` with
  the same body and return annotation. A new `_default_store_handler(obj, pool) -> None: obj.store()`
  takes its old name.
- Constructor signature (was `:69-88`): `persist_handler=_default_persist_handler` added
  immediately after `store_handler=_default_store_handler`; every later parameter's position is
  unchanged. `self._persist_handler = persist_handler` added alongside the existing
  `self._store_handler = store_handler`.
- Constructor validation (was `:89-97`): both checks now test `persist_handler` instead of
  `store_handler`; both messages reworded from "store maker"/"store"-only phrasing to name the
  persist maker (`"a persist maker must also be supplied…"`, `"a persist maker was provided…"`),
  while the second message's tail (`"…because there will be no compute results to store"`) is left
  as upstream has it — it describes the *symptom* (nothing to store), not the hook name.
- Status-message builders (were `:167` and `:204`): both `if self._store_handler is not None:`
  guards switched to `if self._persist_handler is not None:`, since it is the persist handler that
  generates the `"store"`-typed work items being counted. The message text itself (`f", {…} store"`)
  is untouched.
- Compute branch (was `:420-431`): `obj.store()` (hardcoded) replaced with
  `self._store_handler(obj, self._pool)`, with the explanatory comment from the prompt added above
  it; `store_task: ObjectRef = self._store_handler(obj, self._pool)` (the persist call, previously
  misnamed via the old single hook) replaced with `self._persist_handler(obj, self._pool)`. The
  stale comment `# is a compute handler was supplied, a store handler must have been also` corrected
  to `# a compute handler was supplied, so a persist handler must have been also` (grammar fix
  bundled with the rename since it is the same line). The `"store"` work-item type string, the
  `_num_store_queue`/`_num_store_complete` counters, and the `_data[...] = ("store", payload)`
  tagging are byte-for-byte unchanged, per the prompt's explicit instruction.

### Task 2 — the 35 call sites

Every `store_handler=None` line across the 7 files got a `persist_handler=None,` line inserted
immediately after it, at matching indentation, via a small Python script (not `sed`, to keep the
indentation-matching explicit and auditable) — not committed, scratchpad only. Distribution
matches the prompt's table exactly:

| File | Sites |
|---|---|
| `main.py` | 19 |
| `extract_Gk_data.py` | 3 |
| `extract_GkSource_data.py` | 3 |
| `extract_GkWKB_data.py` | 3 |
| `extract_QuadSourceIntegral_data.py` | 3 |
| `extract_TkWKB_data.py` | 2 |
| `extract_tensor_source_data.py` | 2 |
| **total** | **35** |

No site needed anything beyond the mechanical insertion — none of the 35 `RayWorkPool(...)`
constructions had an unusual argument order or a comment sitting between `store_handler=None,` and
the following line that would have broken the "insert immediately after" rule.

## Deviations from the prompt

None. `RayWorkPool.py` matches the prompt's code blocks exactly (defaults, constructor, validation
messages re-worded per the prompt's instruction to "check upstream's wording and follow it, or
improve it", status-message guards, compute-branch comment and calls). All 35 call sites received
the mechanical edit with no adaptation required.

## Verification performed

1. **Grep counts (actually run):**
   - `grep -rn "store_handler=None" --include="*.py" . | grep -v venv | wc -l` → **35**, both before
     and after this prompt's edits (the edits only *add* `persist_handler=None` lines; they do not
     touch the `store_handler=None` lines themselves).
   - `grep -rn "persist_handler=None" --include="*.py" . | grep -v venv | wc -l` → **35** (0 before,
     35 after).
   - **Pairing check (actually run, not just counted):** a throwaway Python script read each of the
     7 files, found every line containing `store_handler=None`, and asserted the *immediately
     following* line contains `persist_handler=None`. Zero mismatches across all 35 sites.
   - `grep -rn "RayWorkPool(" --include="*.py" . | grep -v venv | wc -l` → **45**, unchanged.
   - `grep -rn "store_handler=" --include="*.py" . | grep -v venv | grep -v "store_handler=None" |
     grep -v "RayTools/RayWorkPool.py"` → **0 hits**, confirming (per the prompt's instruction to
     verify this claim rather than trust it) that no call site anywhere passes a *custom*
     `store_handler`; the only non-`None` occurrences are the definition and default-parameter sites
     inside `RayWorkPool.py` itself.
2. **Compile:** `./venv/bin/python3 -m py_compile` on all 8 touched files — clean, exit 0.
3. **Formatting:** `./venv/bin/python3 -m black --check` on all 8 touched files — "All done! 8 files
   would be left unchanged."
4. **Behavioural (actually run, not just reasoned about):** a throwaway harness
   (`/private/tmp/.../scratchpad/verify_prompt05.py`, not committed) that, against the real
   `RayTools.RayWorkPool` module (imported directly, no stub needed — see note below):
   - Called `_default_store_handler(obj, pool)` then `_default_persist_handler(obj, pool)` against
     fake `obj`/`pool` stand-ins that record call order. Confirmed the two calls fire in the order
     `obj.store()` then `pool.object_store(obj)` — the same order and the same two operations the
     old single hardcoded-`obj.store()` + single-hook design produced — and that
     `_default_persist_handler` returns the pool's return value (`ObjectRef` in real use).
   - Constructed a `RayWorkPool` with `compute_handler` set and `persist_handler=None`: raised
     `RuntimeError` with the reworded persist-maker message.
   - Constructed one with `compute_handler=None` and a real `persist_handler`: raised
     `RuntimeWarning` with the reworded persist-maker message (confirming it really is raised, not
     merely emitted as a warning — Python's built-in `RuntimeWarning` is a normal exception class
     unless a `warnings` filter intercepts it, and none does here since the code uses `raise`).
   - Constructed one with `compute_handler=None, store_handler=None, persist_handler=None` — the
     exact pattern now used at all 35 migrated call sites — and confirmed it constructs without
     raising.
   - Constructed one with every handler left at its default and confirmed it still constructs
     without raising.
5. **Note on the `defaults`/`config.defaults` import issue:** the harness imported
   `RayTools.RayWorkPool` directly (which imports `Datastore.SQL.ShardedPool`, which imports
   `LiouvilleGreen.WKBtools`) with **no `sys.modules` stub**, unlike every verification harness in
   prompts 02–04. This is expected, not a new finding: the `[02-shard-config-reader]` issue was
   resolved out-of-sequence on 2026-09-04 (see `IMPLEMENTATION_STATE.md` §4), before this prompt was
   picked up, so this is simply the first prompt to benefit from that fix. Confirms the fix
   generalises to this import path too.
6. **Not run:** any part of the actual Ray pipeline (`ray.init()` with real actors, a real
   `main.py`/`extract_*.py` invocation). This needs a live multi-process Ray cluster and a populated
   datastore, the same class of gap already open for prompts 01 and 04 — recorded as a new §3 issue
   below for prompt 10.

## Observations not acted on

- None beyond what is already tracked. No new latent faults were noticed in the touched code beyond
  what the prompt already describes.

## State handed to the next prompt

- The backport proper (audit items B1–B5/D1–D4/F3/F6/E1) is now **complete** as of this commit —
  the "After 05" natural stopping point in `README.md` §4 is reached. Everything the audit
  recommended has landed; nothing else has.
- `RayWorkPool`'s three-hook shape (`compute_handler` / `store_handler` / `persist_handler`) is now
  stable. Prompt 06 (`inventory()` plumbing) does not touch `RayWorkPool` at all per its own file
  list (`Datastore.py`, `ShardedPool.py` only), so no interaction is expected, but note for prompt 09
  (the first plausible real consumer of the `store_handler` hook, per this prompt's own header) that
  the hook signature is `store_handler(obj, pool) -> None`, called locally in the driver, and
  `persist_handler(obj, pool) -> ObjectRef`, dispatched as a Ray task — if prompt 09 overrides
  `store_handler` to mint associated datastore objects, it must not attempt to return anything
  meaningful from it (the return value is discarded; only `persist_handler`'s return is treated as
  an `ObjectRef` and queued into `self._inflight`).
- Two behavioural gaps are now open for prompt 10, all in the same class ("the code change is
  verified structurally/synthetically, not against a live Ray pipeline"): the `[01-shard-key-
  persistence]` and `[04-read-table-service]` issues already in `IMPLEMENTATION_STATE.md` §3, plus
  the new `[05-persist-handler-split]` issue this prompt opens (see status board).
