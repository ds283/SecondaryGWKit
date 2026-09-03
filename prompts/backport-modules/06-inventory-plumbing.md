# Prompt 06 — `inventory()` plumbing in `Datastore` and `ShardedPool` (F2, part 1 of 4)

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Audit section:** §4 F2
**Depends on:** prompt 04. `read_table` and `inventory` are near-identical dispatch methods sitting
next to each other in both classes; landing 04 first means you write `inventory` to match a settled
`read_table` rather than to a shape that is about to change.
**Files you may touch:** `Datastore/SQL/Datastore.py`, `Datastore/SQL/ShardedPool.py`, plus the log
and status board. **Not** `config/sharding.py` and **not** the factories — those are prompts 07/08.

---

## Where F2 sits in this campaign

F2 was originally recommended for deferral, because the plumbing is inert without a per-factory
`inventory()` method and `SGWK` has zero of those. **The user has confirmed they want the reporting
capability**, so F2 is in scope and is delivered across four prompts:

| Prompt | Scope |
|---|---|
| **06 (this one)** | `Datastore.inventory`, `ShardedPool.inventory`, `_merge_queue`, `InventoryConfigType`, the `inventory_config` constructor parameter |
| 07 | `inventory()` on the 13 replicated-table factories |
| 08 | `inventory()` on the 15 sharded-table factories, plus the `inventory_config` merge policies |
| 09 | The reporting entry point that actually prints a datastore inventory |

After this prompt the service exists and raises a clear, specific error for every class, because no
factory implements `inventory` yet. **That is the correct end state for this commit.** Do not add
factory methods here to make something work end to end — the split exists so each commit stays
reviewable.

---

## Scoping decisions already made

Three things were settled by the planning pass. You do not need to re-derive them, but the reasoning
matters because it is why `SGWK`'s version differs from upstream's.

### (a) `inventory` must be an optional `@staticmethod`, not an abstract method

`SQLAFactoryBase` (`Datastore/SQL/ObjectFactories/base.py`) is an ABC whose members — `register`,
`build`, `store`, `validate`, `validate_on_startup` — are all `@staticmethod` **and**
`@abstractmethod`. Adding `inventory` as an abstract method would force all ~37 factory classes in
the tree to implement it immediately, including tag-association factories for which an inventory is
meaningless.

So: **do not touch `base.py`.** Discover the method with `hasattr(factory, "inventory")`, exactly as
upstream does, and raise a clear error when it is absent.

This also means the signature differs from upstream. `SI` registers factory *instances* and its
inventory methods are instance methods, `def inventory(self, conn, table, tables)`. `SGWK` registers
factory *classes* with `@staticmethod`s (audit §5 **X4**, which we are deliberately not backporting).
`SGWK`'s factory signature is therefore:

```python
    @staticmethod
    def inventory(conn, table, tables, *args, **kwargs):
```

The call in `Datastore.inventory` is identical either way — `factory.inventory(conn, tab, self._tables, ...)`
binds correctly against both — so only the factory side changes. Prompts 07 and 08 rely on this;
state it in your log so they inherit it.

### (b) `tables` is passed unconditionally, unlike `read_table`

Upstream's `Datastore.inventory` passes `self._tables` positionally and always
(`SI/Datastore.py:814`), with no `tables_arg` switch. Keep that. It is a deliberate difference from
`read_table` (prompt 04), where the switch exists because two pre-existing factory signatures
disagreed. Here every factory method is new, so they can simply all accept `tables`. Simpler, and
one fewer config field to maintain.

### (c) `_merge_queue` needs a numeric policy that upstream does not have

This is the one substantive design addition, and it is forced by `SGWK`'s schema rather than
optional.

Upstream's `_merge_queue` dispatches on the *value* type and supports exactly:

| Type | Policy |
|---|---|
| `list` | `"extend"` |
| `set` | `"extend"` |
| `datetime` | `"earliest"`, `"latest"` |
| `None` | take the other value |

Anything else falls through to `RuntimeError`. The audit flagged this as "worth tightening on the
way in" for a hypothetical `int` count field. **For `SGWK` it is not hypothetical.** Six of the 15
sharded tables are high-volume value tables — `TkNumericValue`, `TkWKBValue`, `QuadSourceValue`,
`GkNumericValue`, `GkWKBValue`, `GkSourceValue` — and all six are registered with
`"timestamp": False`, so they have no timestamp column at all (verified in each factory's
`register()`). The only inventory that makes sense for them is a **row count**, and a count from
each shard has to be **summed**, not extended. There is no way to express that in upstream's policy
set.

So add numeric handling. Suggested:

```python
                elif isinstance(current, (int, float)) and not isinstance(current, bool):
                    if policy == "sum":
                        data[field] = current + next
                        continue
                    elif policy == "min":
                        data[field] = min(current, next)
                        continue
                    elif policy == "max":
                        data[field] = max(current, next)
                        continue
```

`"sum"` is the one prompt 08 actually needs; `"min"` / `"max"` are cheap and obvious companions —
include them or don't, but say which and why. Note the `bool` exclusion: `bool` is a subclass of
`int` in Python, and silently summing booleans would be a confusing way to get a wrong answer.

Guard the `None` values too. A shard that holds no rows for a class will return `None` for its
timestamp fields, and possibly `0` or `None` for a count. Upstream's `elif current is None:` branch
handles a `None` *first* value, but the `datetime` branch's `next is not None` guards are what stop
a `None` *later* value from breaking the comparison — make sure your numeric branch is equally
defensive about `next` being `None`, and decide deliberately what `sum` should do with it.

---

## Task 1 — `Datastore.inventory`

Add alongside `read_table` in `Datastore/SQL/Datastore.py`, modelled on `SI/Datastore.py:789-816`:

```python
    def inventory(self, cls, *args, **kwargs):
        """
        Return a human-readable inventory of the Datastore contents for a particular object class
        :return:
        """
        if isinstance(cls, str):
            class_name = cls
        else:
            class_name = cls.__name__

        with ProfileBatchManager(
            self._profile_batcher, f"inventory[{class_name}]"
        ) as mgr:
            self._ensure_registered_schema(class_name)
            record = self._schema[class_name]

            tab = record["table"]
            factory = self._factories[class_name]

            if not hasattr(factory, "inventory"):
                raise RuntimeError(
                    f'Datastore: the object factory for "{class_name}" does not provide an inventory service'
                )

            with self._engine.begin() as conn:
                objects = factory.inventory(conn, tab, self._tables, *args, **kwargs)

            return objects
```

Match whatever `ProfileBatchManager` convention prompt 04 settled on for `read_table` — including
the return-inside-versus-outside-the-`with` decision. **The two methods must agree**; if they
disagree, a later reader cannot tell which is intentional.

Add `InventoryConfigType = Dict[str, Any]` next to `ReadTableConfigType`
(cf. `SI/Datastore.py:154`).

Note there is no `inventory_config` parameter on `Datastore` — only on `ShardedPool`. The merge
policy is a cross-shard concern, so a single `Datastore` never needs it.

## Task 2 — `ShardedPool._merge_queue`

Add, modelled on `SI/ShardedPool.py:882-930`, with the numeric policy from (c) above.

Two things to fix on the way in, neither of which upstream got right:

1. **`merge_queue.pop()` mutates the caller's list.** Upstream starts `data = merge_queue.pop()` and
   iterates the remainder. Since the caller passes a freshly-built `ray.get(...)` list this is
   currently harmless, but it makes the function unsafe to reuse and is gratuitous. Take a copy, or
   index rather than pop. It also means the *last* shard's dict becomes the accumulator and is
   mutated in place — fine, but worth a comment.
2. **The `"extend"` policy mutates `current` in place** (`current.extend(next)`), which writes
   through into the shard's returned dict. Again harmless today, but decide deliberately and say so.

Keep upstream's error messages — they are specific and useful — and add one for the numeric case.

## Task 3 — `ShardedPool.inventory` and `inventory_config`

Constructor: add `inventory_config: Optional[InventoryConfigType] = None` to the signature (upstream
puts it next to `read_table_config`, `SI/ShardedPool.py:54`) and store it:

```python
        self._inventory_config: Optional[InventoryConfigType] = inventory_config
```

Import `InventoryConfigType` alongside `ReadTableConfigType` from `Datastore.SQL.Datastore`.

Method, modelled on `SI/ShardedPool.py:932-1001`:

- **Replicated class** → pick one shard at random, `ray.get(shard.inventory.remote(class_name, *args, **kwargs))`.
  Note upstream `ray.get`s here, so `inventory` returns a **value**, not an `ObjectRef` — unlike
  `read_table`, which returns the ref. That asymmetry is upstream's and is reasonable (the merge path
  has to resolve anyway, so both branches returning values keeps the caller uniform). Keep it, and
  note it in the log so prompt 09 is not surprised.
- **Sharded class** → require `inventory_config`, require the class to be in it, fan out to every
  shard, then merge.
- **Neither** → `raise RuntimeError(f'Unable to dispatch inventory() for item of type "{class_name}"')`.

The merge branch detects the return shape by inspecting the first field:

```python
            field = list(data_queue[0].keys()).pop()
            if isinstance(data_queue[0][field], dict):
                # labelled form: {label: {field: value}}
            else:
                # flat form: {field: value}
```

**Two corrections to make to this, relative to both upstream and the audit.**

*The audit's stated defect here is wrong.* Audit §4 F2 says `SI`'s `ShardedPool.inventory` "calls
`self._inventory_config[class_name]` without first checking `self._inventory_config is not None` in
the label-merge branch". It does check — `SI/ShardedPool.py:958` raises when the config is `None`,
before either branch runs. Do not add a redundant guard on the strength of the audit's claim; verify
for yourself and record the correction in your log so the campaign's record is accurate.

*There is a real defect nearby, though.* In the labelled branch, `self._inventory_config[class_name][label]`
will `KeyError` if a factory returns a label the config does not mention. That produces a bare
`KeyError: 'validated'` rather than a diagnosis. Raise a proper message naming the class, the label,
and the labels the config does know about. Prompt 08 has to keep ~9 labelled configs in step with
~9 factory methods by hand, so this error is one somebody will actually hit.

Also consider: the shape sniff takes an arbitrary field via `list(...).pop()` (which is the *last*
key, since `pop()` with no argument pops the end) and assumes every other field agrees. A factory
returning a mix of dict and non-dict values at the top level would be misread. A cheap assertion
that all top-level values agree in kind is worth adding; your call, but say which way you went.

---

## Do not

- Do not modify `Datastore/SQL/ObjectFactories/base.py`. See (a).
- Do not add `inventory()` to any factory. Prompts 07 and 08.
- Do not add `inventory_config` to `config/sharding.py`. Prompt 08 — it has to be written against
  the actual factory return shapes, which do not exist yet.
- Do not backport X1, X2, X3, X4 (audit §5).

---

## Verification

1. Both files parse; `black --check` clean if available.
2. `Datastore.inventory` and `Datastore.read_table` agree on their `ProfileBatchManager` convention.
3. **Exercise `_merge_queue` directly.** It is a pure function of its arguments — no Ray, no
   database — so it is the one part of this commit that is straightforwardly unit-testable. Write a
   throwaway harness (scratchpad, not committed) covering at minimum:
   - flat merge across three shard dicts with `extend` / `earliest` / `latest` / `sum`;
   - a shard returning `None` timestamps (the empty-shard case);
   - a field missing from the config → the intended `RuntimeError`, not a `KeyError`;
   - an unknown policy for a known type → the intended `RuntimeError`;
   - `bool` values are not silently summed.

   Record the actual output. This is the highest-value verification available in this prompt and
   there is no excuse for skipping it.
4. `pool.inventory("wavenumber")` raises the "does not provide an inventory service" error, and
   `pool.inventory("GkSource")` raises the "not configured" error. Both are expected failures at
   this point in the campaign — confirm the *message* is the intended one, since a stray
   `AttributeError` or `KeyError` here would be indistinguishable from a real fault later.

---

## Finish

1. Write `prompts/backport-modules/logs/06-inventory-plumbing.md` per `README.md` §5.1. Record: the
   numeric-policy design and exactly which policies you added; the `_merge_queue` mutation
   decisions; the correction to the audit's F2 claim; the `@staticmethod` signature contract that
   prompts 07/08 depend on; and the `ray.get`-in-`inventory` asymmetry that prompt 09 depends on.
   The "State handed to the next prompt" section matters more here than in any other prompt —
   07 and 08 are written against contracts you are fixing.
2. Update `IMPLEMENTATION_STATE.md`: the prompt 06 row, the F2a item row, progress, "last updated",
   any §3 issues.
3. Commit. Suggested message:

```
Add an inventory() service to Datastore and ShardedPool

First of four commits adding datastore-contents reporting. This one is the
pool and datastore plumbing only: no factory implements inventory() yet, so
every class raises a clear "does not provide an inventory service" error
until the next two commits land.

Datastore.inventory dispatches to the object factory, discovering the method
with hasattr rather than making it abstract on SQLAFactoryBase -- an abstract
method would oblige all ~37 factory classes to implement one, including tag
association factories where an inventory is meaningless. Because this tree
registers factory classes with staticmethods rather than instances, the
factory signature is (conn, table, tables, *args, **kwargs).

ShardedPool.inventory queries one shard for a replicated class and fans out
to every shard for a sharded one, merging the per-shard dicts under a
declarative per-field policy. Both the flat {field: value} and the labelled
{label: {field: value}} return shapes are supported.

_merge_queue gains numeric policies, which upstream does not have. Six of the
15 sharded tables are value tables registered with "timestamp": False, so
their only meaningful inventory is a row count, and counts have to be summed
across shards rather than extended. Upstream's policy set covers only lists,
sets, datetimes and None, and would raise for an int.

Also raises a diagnosable error when a factory returns a label the merge
configuration does not mention, rather than a bare KeyError.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```
