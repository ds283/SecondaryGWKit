# Log 06 — `inventory()` plumbing in `Datastore` and `ShardedPool` (F2a)

**Prompt:** prompts/backport-modules/06-inventory-plumbing.md
**Commit:** *(SHA intentionally not embedded — see `IMPLEMENTATION_STATE.md` §5 note 11)*
**Date:** 2026-09-04
**Result:** COMPLETE

## What shipped

### Task 1 — `Datastore.inventory` (`Datastore/SQL/Datastore.py`)

- Added `InventoryConfigType = Dict[str, Any]` immediately after `ReadTableConfigType` (`Datastore.py`,
  just above the `Datastore` class), with a comment noting it maps `class_name -> {field_name ->
  merge policy}` and is consumed only by `ShardedPool`.
- Added `Datastore.inventory(self, cls, *args, **kwargs)` immediately after `read_table`. Follows
  `read_table`'s exact shape: resolve `class_name`, open a `ProfileBatchManager(...,
  f"inventory[{class_name}]")` block, `_ensure_registered_schema`, look up `record`/`tab`/`factory`,
  `hasattr(factory, "inventory")` guard raising `RuntimeError('Datastore: the object factory for
  "{class_name}" does not provide an inventory service')`, then `factory.inventory(conn, tab,
  self._tables, *args, **kwargs)` inside `with self._engine.begin() as conn:`, and `return objects`
  **after** the `ProfileBatchManager` block closes — matching `read_table`'s return-outside-`with`
  convention (itself matching `object_get`/`object_validate`'s pre-existing local convention, per
  prompt 04's log). `tables` is passed unconditionally, per the prompt's scoping note (b) — no
  `tables_arg`-style switch, since every `inventory` factory method is new and can simply all accept
  it.

### Task 2 — `ShardedPool._merge_queue` (`Datastore/SQL/ShardedPool.py`)

Added as a `@staticmethod` (no shard/instance state is needed) directly above `inventory`. Dispatches
on the type of the *current* (accumulator) value:

- `list` → policy must be `"extend"`; `current.extend(next_value)` (mutates the accumulator, and
  therefore the underlying per-shard dict, in place — see mutation notes below).
- `set` → policy must be `"extend"`; `current.update(next_value)` (upstream's own `.extend()` call
  would `AttributeError` on a real `set`, since sets have no `.extend` method — this is a structural
  correction, not a stylistic one: `.update()` is the set equivalent of the "extend" policy).
- `datetime` → policy `"earliest"` (`min`) or `"latest"` (`max`); anything else raises.
- `int`/`float`, **excluding `bool`** → policy `"sum"`, `"min"`, or `"max"`; anything else raises.
  This is the substantive addition the audit calls for (§2.3(b) of the README) — `"sum"` is the one
  prompt 08 actually needs (six value tables with `"timestamp": False` have no meaningful inventory
  beyond a summed row count); `"min"`/`"max"` were included alongside it as cheap, obvious companions
  with the same shape as the datetime pair, on the basis that a numeric field with only `"sum"`
  available and no way to ask "what's the smallest/largest shard's count" would be an arbitrary
  restriction with no cost to lifting.
- Anything else (a value type with no defined merge semantics) → `RuntimeError` naming the field,
  class, and the value's type name.

**`bool` guard, explicit rather than incidental.** The prompt warns that `bool` is a subclass of
`int` in Python and would otherwise be silently caught by the numeric branch. Rather than relying on
ordering (`isinstance` checks for `list`/`set`/`datetime` happening not to catch `bool`, then falling
through the numeric branch to the final catch-all `RuntimeError`), added an explicit `isinstance(...,
bool)` check *before* the numeric branch that raises a dedicated, more specific message ("do not know
how to merge boolean field") rather than the generic catch-all's ("value type bool") — a deliberate
implementation choice for a clearer diagnosis, verified in check 5 below.

**`None` handling**, applied uniformly ahead of all type-specific branches: if the accumulator's
current value is `None`, take the next shard's value outright (mirrors upstream's `None`-as-first-value
handling); if the *next* shard's value is `None` (the empty-shard case — a shard with no rows for a
sharded class returns `None` for a timestamp field and possibly `None` for a count), the accumulated
value is left unchanged rather than merged. **Deliberate choice for `sum`:** a `None` count from an
empty shard contributes nothing and is skipped, rather than being treated as `0` (which would have the
same numeric effect for `sum` but would raise for `min`/`max` if the *only* value seen were `None`) or
raising (which would make an empty shard a hard failure for a service whose whole point is to report
across shards that may legitimately hold nothing for a given class). Verified in check 2 below.

**Two mutation issues, fixed as instructed rather than left as upstream has them:**

1. `merge_queue.pop()` mutating the caller's list → the function now does `queue = list(merge_queue)`
   before popping, so the caller's own list object is untouched (verified in check 8 below). The
   *popped dict itself* is not deep-copied, so it becomes the accumulator and is mutated in place —
   this is unchanged from upstream and is fine given the only caller builds a fresh list from
   `ray.get(...)` each time; documented with an inline comment rather than fixed further, since a
   defensive copy of every per-shard dict would be pure overhead for no observed benefit.
2. `"extend"` mutating `current` in place (`current.extend(...)` / `current.update(...)`), which
   writes through into the accumulator dict that came from one specific shard's `ray.get(...)` result
   — kept as upstream does it (harmless under the current call pattern, since nothing else reads that
   per-shard dict after the merge), with an inline comment flagging it for any future caller that
   reuses per-shard results after merging.

### Task 3 — `ShardedPool.inventory` and `inventory_config`

- Constructor (`ShardedPool.__init__`): added `inventory_config: Optional[InventoryConfigType] =
  None` to the signature, immediately after `read_table_config` (matching upstream's placement per
  the prompt). Stored unconditionally as `self._inventory_config: Optional[InventoryConfigType] =
  inventory_config` after the existing `read_table_config` validation block. **No analogous
  constructor-time validation loop was added** (unlike `read_table_config`'s "must be a replicated
  table" check) — the prompt's Task 3 text describes only "add to the signature ... and store it",
  with no equivalent constructor-time check requested, and `inventory`'s own dispatch already
  validates class membership against `_inventory_config` at call time (see below). Added as a
  deliberate scope decision, not an oversight — see "Observations not acted on" for the option this
  leaves on the table.
- Imported `InventoryConfigType` alongside `ReadTableConfigType` from `Datastore.SQL.Datastore`, and
  added `from datetime import datetime` (needed by `_merge_queue`, not previously imported in this
  file).
- `ShardedPool.inventory(self, cls, *args, **kwargs)`, added directly after `read_table`:
  - **Replicated class:** pick one shard at random (identical random-swap-and-pop pattern already used
    by `read_table`/`_store_impl_replicated_table`/`_validate_impl_replicated_table` in this file), and
    `return ray.get(shard.inventory.remote(class_name, *args, **kwargs))` — a **value**, not an
    `ObjectRef`, per the prompt's explicit note that this is a deliberate asymmetry with `read_table`.
  - **Sharded class:** require `self._inventory_config is not None`
    (`"ShardedPool: the inventory service is not configured"`), require `class_name in
    self._inventory_config` (`"...is not available for objects of class ..."`), then
    `ray.get([shard.inventory.remote(...) for shard in self._shards.values()])` to fan out to every
    shard, then merge.
  - **Neither:** `raise RuntimeError(f'Unable to dispatch inventory() for item of type
    "{class_name}"')` — same message shape as `object_store`/`object_validate`'s existing dispatch
    `RuntimeError`s in this file.
  - **Shape sniff:** `field = list(data_queue[0].keys()).pop()` / `labelled =
    isinstance(data_queue[0][field], dict)`, exactly as the prompt's code sketch. **Added the
    suggested cheap assertion:** `any(isinstance(value, dict) != labelled for value in
    data_queue[0].values())` raises a `RuntimeError` naming the class if the first shard's own
    top-level dict mixes labelled and flat fields — catches the exact failure mode the prompt
    describes (the sniff assumes every field agrees in kind) at the point of the sniff itself, rather
    than producing a confusing downstream `KeyError`/`TypeError` in the merge.
  - **Labelled branch:** for each label in `data_queue[0].keys()`, look up `field_config[label]` (the
    per-label field→policy sub-dict), raising a diagnosable `RuntimeError` naming the class, the
    missing label, and the labels the config *does* know about
    (`sorted(field_config.keys())`) if the label is absent — this is the fix for the real defect
    described in the prompt (a bare `KeyError` today would become a `KeyError` tomorrow once prompt 08
    adds labelled factories). Then `_merge_queue(class_name, [d[label] for d in data_queue],
    field_config[label])` per label, collected into `merged: Dict[str, Dict]` and returned.
  - **Flat branch:** `return self._merge_queue(class_name, data_queue, field_config)` directly, where
    `field_config = self._inventory_config[class_name]` is used as the field→policy dict.

## Deviations from the prompt

### Audit correction, confirmed rather than assumed — as instructed

The prompt asks the implementer to verify the audit's claim (that `SI`'s `ShardedPool.inventory`
"calls `self._inventory_config[class_name]` without first checking `self._inventory_config is not
None`") rather than take it on trust, since `SI`'s source is not present in this tree to inspect
directly. This campaign has no access to the `SI`/`StochasticInstantons` source tree — only the audit
document's line-numbered claims about it — so the correction cannot be independently re-verified
against `SI`'s actual source from inside this repository. **Classification: IMPLEMENTATION CHOICE.**
Rather than silently trust either the audit's claim or the prompt's counter-claim, this implementation
follows the prompt's instruction on its merits — guard `self._inventory_config is not None` *before*
either merge branch runs, which is what any correct implementation must do regardless of what `SI`
actually contains — and does not re-assert the audit's specific claim about `SI`'s source as
independently confirmed. The guard is present in `SGWK`'s version either way (see Task 3 above:
`self._inventory_config is None` is checked immediately on entering the sharded-class branch, before
the labelled/flat sniff or either merge path).

### Numeric merge policy set — IMPLEMENTATION CHOICE

Included `"sum"`, `"min"`, and `"max"` (the prompt explicitly leaves this open: "include them or
don't, but say which and why"). Reasoning given above under Task 2. No `"extend"`-for-numeric or other
policy was added — a numeric field only ever needs one of these three for any table in this schema
(row counts need `"sum"`; nothing in the `config/sharding.py` schema needs a numeric min/max today, but
prompt 08 is the one that will actually populate `inventory_config`, so this prompt does not know yet
whether `"min"`/`"max"` will ever be used — they cost nothing to include speculatively here since the
whole point of `_merge_queue` is to be a generic, reusable dispatcher, not one hand-fitted to today's
known field set).

### No other deviations

Both `Datastore.inventory` and `ShardedPool.inventory` were implemented to the prompt's code sketches
without structural changes; the `_merge_queue` design follows Task 2's guidance exactly except for the
`set`/`"extend"` correction noted above (a bug in the naive transcription of upstream's approach, not a
deviation from what the prompt asked for — the prompt's own suggested numeric-policy snippet was taken
near-verbatim, and the `set` case needed `.update()` for the code to actually run, which `black`/
`py_compile` alone would not have caught since `.extend` and `.update` are both valid Python — only the
runtime check in verification item 3 below caught it).

## Verification performed

1. **Both files parse; `black --check` clean.** `./venv/bin/python3 -m py_compile
   Datastore/SQL/Datastore.py Datastore/SQL/ShardedPool.py` — exit 0.
   `./venv/bin/python3 -m black ShardedPool.py` (needed one reformat pass on first write, then clean)
   and `black --check` on both files — "2 files would be left unchanged."
2. **`Datastore.inventory` and `Datastore.read_table` agree on their `ProfileBatchManager`
   convention.** Confirmed by inspection: both open `ProfileBatchManager(self._profile_batcher,
   f"<name>[{class_name}]")`, assign the factory call's result to `objects` inside the block, and
   `return objects` after the block closes (outside the `with`).
3. **`_merge_queue` exercised directly** (throwaway harness,
   `scratchpad/verify_merge_queue.py`, not committed), actually run under `./venv/bin/python3`:
   - Flat merge across three shard dicts with `extend` (list), `extend` (set), `earliest`, `latest`,
     `sum` — correct merged result (`ids` from all three shards, `tags` unioned, correct min/max
     datetime, `count` summed to 10).
   - A shard returning `None` timestamps/count (the empty-shard case), tested with the empty shard
     first *and* second in the merge order — correct result both ways (`None` never overwrites a real
     value, in either position).
   - A field missing from the config → confirmed `RuntimeError` (not `KeyError`) with a message naming
     the field, class, and the (empty) list of configured fields.
   - An unknown policy string for a known type (`"bogus"` on a numeric field) → confirmed
     `RuntimeError` naming the policy, field, type, and class.
   - `bool` values are not silently summed → confirmed `RuntimeError` from the dedicated bool guard,
     not a silently-wrong integer sum (`True + False` would otherwise have "worked" and returned `1`).
   - Additionally checked (beyond the prompt's minimum list): `min`/`max` numeric policies produce
     correct results, and the caller's own list object is provably unmutated after a `_merge_queue`
     call (`caller_list == caller_list_copy` after the call).
   - All 8 checks printed their actual output/exception message and passed; script output captured
     above inline in the relevant subsections.
4. **`pool.inventory(...)` error-path checks — actually run, not reasoned about**, via a second
   throwaway harness (`scratchpad/verify_inventory_dispatch.py`, not committed):
   - `Datastore.inventory("wavenumber")`, against a real (non-Ray-wrapped) `Datastore` instance
     obtained via `Datastore.SQL.Datastore.__ray_metadata__.modified_class(...)` (same technique
     prompt 04's log documents, adjusted for the package-quirk correction it also documents — `import
     Datastore.SQL.Datastore as DS_mod` binds `DS_mod` to the actor class itself, not the module, so
     `__ray_metadata__` is read directly off `DS_mod`, not `DS_mod.Datastore`), against a fresh,
     empty SQLite file with no `read_table_config`/`inventory_config` — raised exactly
     `RuntimeError: Datastore: the object factory for "wavenumber" does not provide an inventory
     service`. This import required **no `sys.modules` stub** for `config.defaults`/`defaults` — the
     now-resolved `[02-shard-config-reader]` fix (commit `d6f4a43`) is confirmed to generalise to this
     prompt's verification too, consistent with prompt 04's log recording it as "still needed" at the
     time and `IMPLEMENTATION_STATE.md` §4 recording it as closed since.
   - `ShardedPool.inventory("GkSource")` against a minimally-constructed `ShardedPool`
     (`ShardedPool.__new__(ShardedPool)` with only `_replicated_tables`, `_sharded_tables`, and
     `_inventory_config` populated by hand — the same fallback prompt 04's log used for
     `ShardedPool.read_table`'s negative cases, needed here too since a full `ShardedPool` requires a
     live multi-shard Ray cluster) with `_inventory_config = None` — raised exactly `RuntimeError:
     ShardedPool: the inventory service is not configured`.
   - The same pool with `_inventory_config = {"TkNumericValue": {...}}` (i.e. configured, but without
     `"GkSource"`) — `pool.inventory("GkSource")` raised exactly `RuntimeError: ShardedPool: the
     inventory service is not available for objects of class "GkSource"`.
   - `pool.inventory("NotARealClass")` (neither replicated nor sharded) — raised exactly
     `RuntimeError: Unable to dispatch inventory() for item of type "NotARealClass"`, confirming the
     dispatch fallthrough the prompt does not explicitly list as a check but which Task 3 requires.
   - All four messages are the intended `RuntimeError`s with the intended text — no stray
     `AttributeError`/`KeyError` from an incomplete guard.
   - Scratch database file removed after the run; both harness scripts left in the scratchpad
     directory (not committed), per instructions.
5. **Not run / not applicable at this stage:** the labelled-branch merge path (`self._merge_queue`
   called once per label, with the missing-label diagnostic) has no factory to exercise it against
   yet — zero factories implement `inventory` in this tree until prompt 07/08 land, so this path is
   verified only by code inspection against the prompt's description, not by a live call. This is the
   expected state after this prompt (see README.md's "inert until 07/08" framing) and is not a gap
   opened against this prompt's own scope.

## Observations not acted on

- **No constructor-time validation that `inventory_config`'s keys are all sharded classes**, unlike
  `read_table_config`'s existing "must be a replicated table" loop. `inventory` for a *replicated*
  class never consults `self._inventory_config` at all (it goes straight to a single shard's `.remote`
  call), so an `inventory_config` entry for a replicated class name would simply be silent dead
  configuration rather than a hard error — the opposite failure mode from a missing sharded entry
  (which fails loudly and correctly at call time, per Task 3's dispatch checks). Left alone because the
  prompt's Task 3 text does not ask for this check, and prompt 08 is the one that actually populates
  `inventory_config` in `config/sharding.py` — if prompt 08 wants this guard, it can add a validation
  loop mirroring `read_table_config`'s at that point, once the real config content and shape exists
  to validate against.
- **`data_queue[0]` is used as the sole reference for both the shape sniff and the labelled branch's
  label set** (`for label in data_queue[0].keys()`), matching the prompt's own code sketch. No
  cross-shard consistency check exists for whether *every* shard's dict has the same top-level keys
  (only within a single shard's dict, per the "mixes labelled and flat" assertion added above). A
  factory whose `inventory` method returns a different key set on an empty shard than a populated one
  would silently use whichever shard happens to be `data_queue[0]` to determine the label set, and
  `label_queue = [shard_data[label] for shard_data in data_queue]` would then `KeyError` on any shard
  missing that label — a bare `KeyError`, not a diagnosed one. Not fixed here: no factory exists yet to
  observe this failure mode against, and adding speculative defensive code for a shape no real factory
  has yet produced risks guessing the wrong invariant. Flagged for prompts 07/08: **every `inventory`
  factory method for a table with a `validated` column (audit's labelled-shape tables, per
  `IMPLEMENTATION_STATE.md` §5 note 9) must return the same set of labels regardless of whether the
  shard holds zero or many matching rows**, or this merge path will surface a raw `KeyError` instead of
  the diagnosed one this prompt added for the "label not in config" case.

## State handed to the next prompt

- **The `@staticmethod` factory signature contract, confirmed and unchanged from the prompt's
  statement:** `factory.inventory(conn, tab, self._tables, *args, **kwargs)` is called identically
  whether the factory is a class with `@staticmethod`s (as `SGWK` has throughout) or an instance with
  instance methods — so prompts 07/08 must write `@staticmethod def inventory(conn, table, tables,
  *args, **kwargs):` on every factory, **not** `def inventory(self, conn, table, tables, ...)`. A
  stray `self` will bind `conn`'s argument slot to the class instead and produce a confusing failure
  far from its cause, exactly as `IMPLEMENTATION_STATE.md` §5 note 5 warns.
- **`tables` is passed positionally and unconditionally** — every factory's `inventory` method must
  accept a third positional parameter named `tables` (the full `self._tables` mapping), unlike
  `read_table`'s conditional `tables_arg`-gated injection. Do not port the `tables_arg` mechanism into
  `inventory_config` in prompt 08 — it does not exist for `inventory` and is not needed, since every
  `inventory` factory method is new and can simply always accept the parameter.
- **`ShardedPool.inventory`'s replicated-class branch calls `ray.get(...)` and returns a value, not an
  `ObjectRef`** — this is the one place `inventory`'s calling contract differs from `read_table`'s
  (`read_table`'s replicated-class branch returns the raw `ObjectRef`, letting the caller `ray.get` it
  lazily). Prompt 09 (the reporting entry point) can therefore call `pool.inventory(class_name, ...)`
  directly for both replicated and sharded classes and get a plain dict/labelled-dict back either way,
  with no `ray.get` of its own required — the pool has already resolved everything internally.
- **The merge-policy vocabulary a factory's `inventory` method's return values must satisfy**, for
  every field it returns on a sharded class: `list`/`set` → `"extend"`; `datetime` → `"earliest"` /
  `"latest"`; `int`/`float` (never `bool`) → `"sum"` / `"min"` / `"max"`. Any other Python type
  returned for a sharded-class field will hit the generic "do not know how to merge" `RuntimeError` at
  merge time — prompt 08's per-factory methods and the `inventory_config` entries it writes in
  `config/sharding.py` must agree, field for field, with this vocabulary and with each other (a
  mismatched policy/type pairing raises at call time, not at import time, exactly as
  `IMPLEMENTATION_STATE.md`'s closing note on F2 scope warns for a half-configured class).
- **The labelled-shape contract**: a factory returning the labelled form must use `{label: {field:
  value}}` at the top level, with every label the factory can ever return present as a key in that
  class's `inventory_config` entry (i.e. `inventory_config[class_name][label]` must resolve for every
  label — see the "Observations not acted on" cross-shard caveat above for the one sharp edge here).
- `IMPLEMENTATION_STATE.md` §5 note 11's no-self-referential-SHA convention followed: no SHA embedded
  in this log's header or in the status board's prompt-06 row.
