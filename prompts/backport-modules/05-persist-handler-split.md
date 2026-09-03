# Prompt 05 — Split `store_handler` into `store_handler` + `persist_handler` (E1)

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Audit sections:** §4 E1, §8 checklist item 7
**Depends on:** prompt 04 (soft). Both touch the same six `extract_*.py` scripts; doing the semantic
change first means this mechanical edit lands on settled call sites and a revert of this commit does
not disturb prompt 04.
**Files you may touch:** `RayTools/RayWorkPool.py`, `main.py`, and the six `extract_*.py` scripts,
plus the log and status board.

---

## Status: confirmed wanted

E1 was flagged in the audit as discretionary — there is no *current* `SGWK` consumer for the split,
so a minimal-risk backport could have skipped it. **The user has confirmed they want it.** Do not
re-litigate the decision or ask again; implement it.

It is the only **source-incompatible** change in the campaign: 35 call sites across 7 files must
change in the same commit or the tree is broken.

Note that prompt 09 (the inventory reporting entry point) is a plausible first real consumer of the
`store_handler` hook — a caller that mints associated datastore objects between compute and persist
is exactly the case the split exists for. No dependency either way, but it is why E1 stopped being
hypothetical.

---

## What the change is

Upstream (`SI` only — this is the one generic change `SI` made beyond `CPBH` in this module) splits
one hook into two:

| Hook | Runs | Default | Purpose |
|---|---|---|---|
| `store_handler(obj, pool) -> None` | locally in the driver, after compute completes | `obj.store()` | resolve the compute result into the object; overridable so a caller can mint associated datastore objects before serialization |
| `persist_handler(obj, pool) -> ObjectRef` | as a Ray task | `pool.object_store(obj)` | the datastore round-trip |

Today `RayWorkPool` hardcodes `obj.store()` at `RayTools/RayWorkPool.py:414` and calls
`self._store_handler(obj, self._pool)` at line 424. The split makes the hardcoded line an
overridable hook and renames the existing hook to `persist_handler`.

`compute_handler=default`, `store_handler=default`, `persist_handler=default` reproduces current
behaviour exactly, so call sites that use the defaults are unaffected.

---

## Task 1 — `RayWorkPool`

Reference: `SI/RayTools/RayWorkPool.py` lines 79, 83, 94–95, 109, 114, 132–133, 188, 225, 446, 456.

**Module-level defaults** (`RayWorkPool.py:64-65`). The existing `_default_store_handler` becomes
`_default_persist_handler`; a new `_default_store_handler` takes its name:

```python
def _default_store_handler(obj, pool) -> None:
    obj.store()


def _default_persist_handler(obj, pool) -> ObjectRef:
    return pool.object_store(obj)
```

Note the return types differ — `store_handler` returns `None`, `persist_handler` returns an
`ObjectRef`. Match upstream's annotations.

**Constructor signature** (`RayWorkPool.py:69-87`). Add `persist_handler=_default_persist_handler`
immediately after `store_handler=_default_store_handler`, so the two stay adjacent and the
positional order of every later parameter is unchanged. Store both:
`self._store_handler = store_handler`, `self._persist_handler = persist_handler`.

**Constructor validation** (`RayWorkPool.py:88-97`). Both checks switch from `store_handler` to
`persist_handler`:

```python
        if compute_handler is not None and persist_handler is None:
            raise RuntimeError(...)

        if compute_handler is None and persist_handler is not None:
            raise RuntimeWarning(...)
```

Update the message text so it refers to a persist handler rather than a "store maker" — check
upstream's wording and follow it, or improve it, but do not leave messages that name the wrong hook.

**Status-message builders** (`RayWorkPool.py:164-176` and `201-213`). Both test
`if self._store_handler is not None:` to decide whether to report the store queue/completion counts.
Both switch to `self._persist_handler`, because it is the persist handler that generates the
`"store"`-typed work items being counted.

**The compute branch** (`RayWorkPool.py:409-430`). Two changes:

```python
                    idx, obj = payload

                    # call the store handler to resolve the Ray future and populate the object;
                    # the default simply calls obj.store(), but this can be overridden (e.g. to
                    # mint associated datastore objects before the persist step)
                    self._store_handler(obj, self._pool)

                    ...

                    # a compute handler was supplied, so a persist handler must have been also
                    store_task: ObjectRef = self._persist_handler(obj, self._pool)
```

i.e. the hardcoded `obj.store()` at line 414 becomes `self._store_handler(obj, self._pool)`, and the
call at line 424 becomes `self._persist_handler(...)`. Leave the `"store"` work-item type string,
the `_num_store_queue` / `_num_store_complete` counters and the `_data[...] = ("store", payload)`
tagging alone — upstream did not rename those, and renaming them would ripple into the `"store"`
branch and the status messages for no benefit.

## Task 2 — the 35 call sites

This is the migration hazard, and it is why the change has to be atomic.

35 sites across 7 files pass `store_handler=None` to mean *"do not persist"*. Under the new
semantics those calls leave `persist_handler` at its default, so the constructor branch
`compute_handler is None and persist_handler is not None` fires `raise RuntimeWarning(...)` — which
does raise, so the failure is loud and immediate rather than silent. That is a mercy, not a licence
to leave any behind.

**Every one of the 35 must pass `persist_handler=None` alongside `store_handler=None`**, matching
how `SI`'s own drivers do it (`SI/main.py:232-233`, `SI/plot_GradientCoupledSolutions.py:943-944`).

Distribution verified at `79f0360`:

| File | `store_handler=None` sites |
|---|---|
| `main.py` | 19 |
| `extract_Gk_data.py` | 3 |
| `extract_GkSource_data.py` | 3 |
| `extract_GkWKB_data.py` | 3 |
| `extract_QuadSourceIntegral_data.py` | 3 |
| `extract_TkWKB_data.py` | 2 |
| `extract_tensor_source_data.py` | 2 |
| **total** | **35** |

For context, there are **45** `RayWorkPool(...)` constructions in total; the other 10 use the
default handlers and need no change.

`store_handler=` appears nowhere else in the tree except the `RayWorkPool` definition itself — there
are no call sites passing a *custom* store handler, so every occurrence is either the literal
`store_handler=None` or absent. That makes the edit mechanical, but **verify that claim yourself**
before relying on it:

```bash
grep -rn "store_handler=" --include="*.py" . | grep -v venv | grep -v "store_handler=None"
```

should show only `RayTools/RayWorkPool.py`.

---

## Do not

- Do not rename the `"store"` work-item type, the `_num_store_*` counters, or the `"store"` branch.
- Do not change `_default_compute_handler`, `available_handler`, `validation_handler` or
  `post_handler`.
- Do not touch anything from prompts 01–04 in this commit.

---

## Verification

Audit §8 gives the check directly:

1. `grep -rn "store_handler=None" --include="*.py" . | grep -v venv` returns **35** hits, and
   **every one** is accompanied by `persist_handler=None`. Verify the pairing, not just the count —
   a script that checks each `RayWorkPool(` construction and reports any with one but not the other
   is worth writing as a throwaway (scratchpad, not committed).
2. `grep -rn "RayWorkPool(" --include="*.py" . | grep -v venv` still returns **45**.
3. All eight touched files parse; `python -m py_compile` them; `black --check` clean if available.
4. A work pool using the default handlers still stores results exactly as before. The defaults
   compose to the same two operations in the same order (`obj.store()` locally, then
   `pool.object_store(obj)` as a task), so this is arguable statically — but say in the log whether
   you argued it or ran it.
5. If you can run any part of the pipeline, do, and record it. Otherwise open a §3 issue for prompt
   10.

---

## Finish

1. Write `prompts/backport-modules/logs/05-persist-handler-split.md` using the template in
   `README.md` §5.1. Record: the before/after counts for both greps; whether any call site needed
   more than adding `persist_handler=None`; and any call site that resisted the mechanical edit.
2. Update `IMPLEMENTATION_STATE.md`: the prompt 05 row, the E1 item row, the progress count, "last
   updated", and any §3 issues. E1 is confirmed wanted, so a skipped row is not an
   acceptable outcome here — if you could not complete it, mark it ⛔ blocked with the reason.
3. Commit in one commit — the change is source-incompatible and must not be split. Suggested
   message:

```
Split RayWorkPool store_handler into store and persist handlers

RayWorkPool hardcoded obj.store() between compute completing and the
datastore round-trip, so a caller that needs to mint dependent datastore
objects in that window had to do it inside compute(). Make that step an
overridable hook.

store_handler(obj, pool) -> None now runs locally in the driver after compute
completes and defaults to obj.store(), exactly reproducing the previous
hardcoded call. The datastore round-trip moves to persist_handler(obj, pool)
-> ObjectRef, defaulting to pool.object_store(obj) -- the previous
_default_store_handler. Constructor validation and the two status-message
builders now test persist_handler, since it is the persist handler that
generates the "store" work items being counted.

This is source-incompatible: 35 call sites across 7 files passed
store_handler=None to mean "do not persist", which under the new semantics
leaves persist_handler at its default. All 35 now pass persist_handler=None
as well. The failure would have been loud rather than silent -- the
constructor raises when a persist handler is supplied without a compute
handler -- but the whole migration lands in one commit regardless.

Call sites using the default handlers are unaffected: the defaults compose to
the same two operations in the same order.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

Adjust to match what you actually did.
