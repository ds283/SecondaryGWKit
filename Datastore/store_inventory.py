"""
A structured inventory of a **closed** ShardedPool store: what it holds, named by physical labels.

``read_inventory(primary)`` opens the store with ``open_read_only`` (prompt 01) and returns a
``StoreInventory``. For each class it holds ``records``, their ``count``, the earliest and latest
``timestamp`` (for display only: no timestamp is in any record) and ``problems``, a list of named
strings. It reads; it never writes, and it needs no Ray.

**A record** (``Record``) is a small JSON-safe object:

- ``key``: the class's physical identity. Its leaves are canonical (``canonical``), and a reference
  to a parent row is the **digest of that parent's own canonical identity** (``reference_digest``),
  never its serial. The parent itself is a record of its own class. No serial, label, name or
  timestamp is in a key, except where the label *is* the identity (``version``, ``store_tag``,
  ``IntegrationSolver``);
- ``tags``: the sorted tuple of ``store_tag`` labels on the row's ``*_tags`` association rows,
  ``()`` where the class has no association table. The ``Run_<label>`` tag is included;
- ``validated``: the row's flag, or ``None`` where the class has no such column;
- ``value_count``: the number of rows in the class's ``*Value`` table whose parent is this row, or
  ``None`` where the class has none. The value tables are counted, one ``GROUP BY`` per table per
  shard, and never listed.

A computed-values field can be added to ``Record`` later, beside these four and outside ``key``,
without changing what they are.

**Who defines identity.** Each factory owns its record builder, a static
``inventory_records(conn, table, tables, context)`` beside its existing ``inventory()``. It states
its key -- the leaf columns, the parent references and the association and value tables -- and
calls ``read_records`` here, which does the reading. The driver builds classes in dependency order
(``INVENTORY_CLASSES``) and hands each builder, in ``context``, a map from serial to canonical key
and to reference digest for every class already built **on that shard**.

**Floats (decision D1).** ``canonical`` is the one function that turns a leaf into its canonical
form. A float becomes ``float.hex`` of the value **as stored**; an integer, string, boolean or
``None`` is kept as it is. The lookups match floats within 1e-7; the key records the stored bits.

**Shards.** A sharded class is the union of every shard's records. A replicated class is read from
every shard and compared on key, tags, validated flag and value count; its records are the
lowest-serial shard's, and any shard that differs is a named problem (``replicated-divergence``).
Nothing is repaired.

**Named problems.** Each problem string starts with its name:

- ``absent-table``: a table is absent from a shard, and reads there as empty;
- ``incomplete``: a column the key needs is absent from a shard, so none of the class's records
  come from that shard;
- ``replicated-divergence``: a replicated class differs between shards;
- ``duplicate``: two or more records of a class share a key and a tag set. All are kept;
- ``orphan-value``: ``*Value`` rows whose parent row is not on their shard;
- ``orphan-tag``: association rows whose parent row or whose tag is not on their shard;
- ``unresolved-parent``: rows referencing a parent row that cannot be resolved. They are not
  records.

A shard on which a class's table, association table or value table is absent, or whose key
columns are incomplete, is left out of that class's replicated comparison: its absence is already
named, and a second, derived "divergence" would say nothing new.

This module imports only the standard library and ``sqlalchemy`` at module scope, so a factory can
import it from inside its ``inventory_records``. The reader and the factory map are imported inside
``read_inventory``.
"""

import contextlib
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import sqlalchemy as sqla

PathType = Union[str, os.PathLike]

# the classes of the inventory, in dependency order: the leaves, then wavenumber_exit_time and
# BackgroundModel, then the compute targets. A class's parents always come before it. The *Value
# and *_tags tables are not classes here: they are read for their parents' value_count and tags
COMPUTE_TARGETS: Tuple[str, ...] = (
    "TkNumericIntegration",
    "TkWKBIntegration",
    "GkNumericIntegration",
    "GkWKBIntegration",
    "GkSource",
    "GkSourcePolicyData",
    "QuadSource",
    "QuadSourceIntegral",
    "OneLoopIntegral",
)

INVENTORY_CLASSES: Tuple[str, ...] = (
    "version",
    "store_tag",
    "redshift",
    "wavenumber",
    "tolerance",
    "LambdaCDM",
    "QCD_Cosmology",
    "IntegrationSolver",
    "GkSourcePolicy",
    "QuadSourcePolicy",
    "wavenumber_exit_time",
    "BackgroundModel",
) + COMPUTE_TARGETS

# the polymorphic cosmology reference of wavenumber_exit_time and BackgroundModel: a serial with no
# foreign key, whose table is named by the row's cosmology_type (CosmologyModels/model_ids.py)
COSMOLOGY = "cosmology"

# up to this many examples are named in a problem
_EXAMPLES = 5


def cosmology_classes() -> Dict[int, str]:
    """``cosmology_type`` -> the class whose table holds that cosmology (CosmologyModels/model_ids.py)."""
    from CosmologyModels.model_ids import LAMBDACDM_IDENTIFIER, QCD_EOS_IDENTIFIER

    return {LAMBDACDM_IDENTIFIER: "LambdaCDM", QCD_EOS_IDENTIFIER: "QCD_Cosmology"}


# ---------------------------------------------------------------------------------------------
# canonical forms and digests
# ---------------------------------------------------------------------------------------------


def canonical(value: Any) -> Any:
    """
    The canonical form of one leaf (decision D1). **This is the only code that formats a float for
    a key.**

    A float becomes ``float.hex`` of the value as stored. An integer, string, boolean or ``None``
    is kept as it is. Anything else raises ``TypeError``: no other kind of leaf is expected, and a
    silent ``str()`` would make two different values look alike.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return float.hex(value)
    raise TypeError(
        f"canonical(): no canonical form for a leaf of type {type(value).__name__!r} ({value!r})"
    )


def _canonical_tree(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"canonical_json(): a key must be a string, not {k!r}")
            out[k] = _canonical_tree(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_canonical_tree(v) for v in obj]
    return canonical(obj)


def canonical_json(obj: Any) -> str:
    """The one canonical JSON form: every leaf through ``canonical``, keys sorted, no whitespace."""
    return json.dumps(
        _canonical_tree(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def digest(obj: Any) -> str:
    """SHA-256 (hex) of ``canonical_json(obj)``."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def reference_digest(key: Mapping[str, Any], tags: Sequence[str], tagged: bool) -> str:
    """
    The digest by which a child refers to a parent. For a class with no association table it is
    the digest of the parent's key. For a tagged class it covers the key **and** the tag set,
    because two rows of a tagged class can differ only in their tags.
    """
    if tagged:
        return digest({"key": key, "tags": list(tags)})
    return digest(key)


# ---------------------------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=True)
class Record:
    """One work item or configuration row. JSON-safe; ``as_json`` gives its plain form."""

    key: Mapping[str, Any]
    tags: Tuple[str, ...]
    validated: Optional[bool]
    value_count: Optional[int]

    def as_json(self) -> Dict[str, Any]:
        return {
            "key": dict(self.key),
            "tags": list(self.tags),
            "validated": self.validated,
            "value_count": self.value_count,
        }

    def canonical_json(self) -> str:
        return canonical_json(self.as_json())

    def identity(self) -> str:
        """The canonical JSON of the key and tag set: what two duplicates share."""
        return canonical_json({"key": self.key, "tags": list(self.tags)})

    def __hash__(self) -> int:
        return hash(self.canonical_json())


@dataclass(frozen=True)
class Parent:
    """A reference from a key field to a parent row, by the serial in ``column``.

    ``of`` names the parent class, or is ``COSMOLOGY``, in which case ``type_column`` holds the
    ``cosmology_type`` that names it."""

    column: str
    of: str
    type_column: Optional[str] = None


@dataclass(frozen=True)
class ShardContext:
    """What the driver hands a factory's record builder, for one shard.

    ``keys[cls][serial]`` is the canonical key, and ``digests[cls][serial]`` the reference digest,
    of every row of every class already built on this shard."""

    shard: int
    absent_tables: FrozenSet[str]
    absent_columns: Mapping[str, Tuple[str, ...]]
    keys: Mapping[str, Mapping[int, Mapping[str, Any]]]
    digests: Mapping[str, Mapping[int, str]]


@dataclass
class ShardRead:
    """One class read from one shard by ``read_records``."""

    name: str
    shard: int
    rows: List[Tuple[int, Record, Optional[datetime]]]
    problems: List[str]
    comparable: bool
    tagged: bool
    parents: Dict[str, str]


@dataclass(frozen=True)
class ClassInventory:
    """One class of the inventory. ``parents`` maps each key field that references a parent to
    the parent class (``COSMOLOGY`` for the polymorphic cosmology reference); ``tagged`` says
    whether the class has an association table, and so whether its reference digest covers
    tags."""

    name: str
    replicated: bool
    tagged: bool
    parents: Mapping[str, str]
    records: Tuple[Record, ...]
    count: int
    earliest_timestamp: Optional[datetime]
    latest_timestamp: Optional[datetime]
    problems: Tuple[str, ...]


@dataclass(frozen=True)
class StoreInventory:
    """The structured inventory of a store: its primary, its shard serials, and each class in
    dependency order."""

    primary: Path
    shards: Tuple[int, ...]
    classes: Mapping[str, ClassInventory]

    def __getitem__(self, name: str) -> ClassInventory:
        return self.classes[name]

    @property
    def problems(self) -> Tuple[str, ...]:
        return tuple(p for c in self.classes.values() for p in c.problems)

    def digest_of(self, name: str, record: Record) -> str:
        """The digest by which a child of class ``name`` would refer to ``record``."""
        return reference_digest(record.key, record.tags, self.classes[name].tagged)

    def find(self, name: str, reference: str) -> List[Record]:
        """The records of class ``name`` whose reference digest is ``reference``."""
        return [
            r
            for r in self.classes[name].records
            if self.digest_of(name, r) == reference
        ]

    def resolve(self, name: str, record: Record) -> Dict[str, Any]:
        """``record``'s key with every parent reference replaced, recursively, by the parent's
        own resolved key and tags. For display and for checking; the key itself is unchanged.
        """
        cls = self.classes[name]
        out: Dict[str, Any] = {}
        types = cosmology_classes()
        for field, value in record.key.items():
            of = cls.parents.get(field)
            if of is None:
                out[field] = value
                continue
            if of == COSMOLOGY:
                of = types.get(record.key.get("cosmology_type"))
            found = self.find(of, value) if of is not None else []
            if len(found) != 1:
                out[field] = {"class": of, "digest": value, "unresolved": len(found)}
                continue
            parent = found[0]
            out[field] = {
                "class": of,
                "key": self.resolve(of, parent),
                "tags": list(parent.tags),
            }
        return out


def _problem(kind: str, name: str, text: str) -> str:
    return f"{kind}: {name}: {text}"


def _examples(values) -> str:
    listed = sorted(set(values), key=lambda v: (str(type(v)), v))[:_EXAMPLES]
    return ", ".join(str(v) for v in listed)


def read_records(
    conn,
    table: sqla.Table,
    tables: Mapping[str, sqla.Table],
    context: ShardContext,
    *,
    leaves: Sequence[str] = (),
    parents: Optional[Mapping[str, Parent]] = None,
    tags: Optional[Tuple[str, str]] = None,
    values: Optional[Tuple[str, str]] = None,
    validated: bool = False,
) -> ShardRead:
    """
    Read one class's records from one shard, as its factory's ``inventory_records`` specifies.

    - ``leaves``: the key's leaf columns, each a key field of the same name, through ``canonical``;
    - ``parents``: key field -> ``Parent``, each resolved to the parent's reference digest through
      ``context``;
    - ``tags``: ``(association table, its column naming this class's serial)``, or ``None``;
    - ``values``: ``(value table, its column naming this class's serial)``, or ``None``;
    - ``validated``: whether the class has a ``validated`` column.
    """
    name = table.name
    shard = context.shard
    parents = dict(parents or {})
    problems: List[str] = []
    parent_classes = {f: p.of for f, p in parents.items()}

    for field, parent in parents.items():
        needed_classes = (
            tuple(cosmology_classes().values())
            if parent.of == COSMOLOGY
            else (parent.of,)
        )
        for of in needed_classes:
            if of not in context.digests:
                raise RuntimeError(
                    f"read_records(): {name}.{field} references {of}, which has not been built "
                    f"on shard #{shard}; classes must be built in dependency order"
                )

    def _read(rows, comparable):
        return ShardRead(
            name=name,
            shard=shard,
            rows=rows,
            problems=problems,
            comparable=comparable,
            tagged=tags is not None,
            parents=parent_classes,
        )

    if name in context.absent_tables:
        problems.append(
            _problem(
                "absent-table",
                name,
                f"table {name} is absent from shard #{shard}, and reads there as empty",
            )
        )
        return _read([], False)

    absent = set(context.absent_columns.get(name, ()))
    needed = list(
        dict.fromkeys(
            list(leaves)
            + [p.column for p in parents.values()]
            + [p.type_column for p in parents.values() if p.type_column is not None]
        )
    )
    missing = [c for c in needed if c in absent]
    if len(missing) > 0:
        problems.append(
            _problem(
                "incomplete",
                name,
                f"shard #{shard} lacks the key column(s) {', '.join(missing)}, so none of "
                f"its {name} records are read from that shard",
            )
        )
        return _read([], False)

    columns = [table.c.serial] + [table.c[c] for c in needed]
    has_validated = validated and "validated" not in absent
    if has_validated:
        columns.append(table.c.validated)
    has_timestamp = "timestamp" in table.c and "timestamp" not in absent
    if has_timestamp:
        columns.append(table.c.timestamp)

    rows = conn.execute(sqla.select(*columns).order_by(table.c.serial)).all()
    serials = {row.serial for row in rows}
    comparable = True

    # the full tag set of each row, from its own association table, joined inside the shard
    tag_sets: Dict[int, List[str]] = {}
    if tags is not None:
        tag_table_name, parent_column = tags
        if tag_table_name in context.absent_tables or parent_column in set(
            context.absent_columns.get(tag_table_name, ())
        ):
            problems.append(
                _problem(
                    "absent-table",
                    name,
                    f"association table {tag_table_name} is absent from shard #{shard}, and "
                    f"reads there as empty",
                )
            )
            comparable = False
        else:
            tag_table = tables[tag_table_name]
            labels = {
                serial: key["label"]
                for serial, key in context.keys.get("store_tag", {}).items()
            }
            no_parent: List[int] = []
            no_tag: List[int] = []
            for parent_serial, tag_serial in conn.execute(
                sqla.select(tag_table.c[parent_column], tag_table.c.tag_serial)
            ):
                if parent_serial not in serials:
                    no_parent.append(parent_serial)
                elif tag_serial not in labels:
                    no_tag.append(tag_serial)
                else:
                    tag_sets.setdefault(parent_serial, []).append(labels[tag_serial])
            if len(no_parent) > 0:
                problems.append(
                    _problem(
                        "orphan-tag",
                        name,
                        f"{len(no_parent)} {tag_table_name} row(s) on shard #{shard} name a "
                        f"{name} row that is not there; e.g. parent serials "
                        f"{_examples(no_parent)}",
                    )
                )
            if len(no_tag) > 0:
                problems.append(
                    _problem(
                        "orphan-tag",
                        name,
                        f"{len(no_tag)} {tag_table_name} row(s) on shard #{shard} name a "
                        f"store_tag that is not there; e.g. tag serials {_examples(no_tag)}",
                    )
                )

    # the number of value rows under each parent: one GROUP BY, never a listing
    counts: Dict[int, int] = {}
    if values is not None:
        value_table_name, parent_column = values
        if value_table_name in context.absent_tables or parent_column in set(
            context.absent_columns.get(value_table_name, ())
        ):
            problems.append(
                _problem(
                    "absent-table",
                    name,
                    f"value table {value_table_name} is absent from shard #{shard}, and reads "
                    f"there as empty",
                )
            )
            comparable = False
        else:
            value_table = tables[value_table_name]
            parent_col = value_table.c[parent_column]
            orphans: Dict[int, int] = {}
            for parent_serial, n in conn.execute(
                sqla.select(parent_col, sqla.func.count()).group_by(parent_col)
            ):
                if parent_serial in serials:
                    counts[parent_serial] = n
                else:
                    orphans[parent_serial] = n
            if len(orphans) > 0:
                problems.append(
                    _problem(
                        "orphan-value",
                        name,
                        f"{sum(orphans.values())} {value_table_name} row(s) on shard #{shard} "
                        f"name {len(orphans)} {name} row(s) that are not there; e.g. parent "
                        f"serials {_examples(orphans)}",
                    )
                )

    types = cosmology_classes()
    out: List[Tuple[int, Record, Optional[datetime]]] = []
    unresolved: List[Tuple[int, str]] = []
    for row in rows:
        mapping = row._mapping
        key: Dict[str, Any] = {leaf: canonical(mapping[leaf]) for leaf in leaves}
        missing_parent = None
        for field, parent in parents.items():
            of = parent.of
            if of == COSMOLOGY:
                of = types.get(mapping[parent.type_column])
            reference = (
                context.digests.get(of, {}).get(mapping[parent.column])
                if of is not None
                else None
            )
            if reference is None:
                missing_parent = field
                break
            key[field] = reference
        if missing_parent is not None:
            unresolved.append((row.serial, missing_parent))
            continue

        record = Record(
            key=key,
            tags=(
                tuple(sorted(tag_sets.get(row.serial, ()))) if tags is not None else ()
            ),
            validated=(
                (None if mapping["validated"] is None else bool(mapping["validated"]))
                if has_validated
                else None
            ),
            value_count=counts.get(row.serial, 0) if values is not None else None,
        )
        out.append(
            (row.serial, record, mapping["timestamp"] if has_timestamp else None)
        )

    if len(unresolved) > 0:
        fields = sorted({f for _, f in unresolved})
        problems.append(
            _problem(
                "unresolved-parent",
                name,
                f"{len(unresolved)} row(s) on shard #{shard} reference a parent that cannot be "
                f"resolved (field(s) {', '.join(fields)}), and are not records; e.g. serials "
                f"{_examples(s for s, _ in unresolved)}",
            )
        )

    return _read(out, comparable)


# ---------------------------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------------------------


def _combine(
    name: str, replicated: bool, reads: Mapping[int, ShardRead]
) -> ClassInventory:
    problems: List[str] = [p for s in sorted(reads) for p in reads[s].problems]
    first = reads[min(reads)]

    if replicated:
        comparable = [s for s in sorted(reads) if reads[s].comparable]
        contributing = comparable[:1]
        if len(comparable) > 0:
            base = comparable[0]
            base_set = Counter(r.canonical_json() for _, r, _ in reads[base].rows)
            base_keys = {r.canonical_json(): r for _, r, _ in reads[base].rows}
            for other in comparable[1:]:
                other_set = Counter(r.canonical_json() for _, r, _ in reads[other].rows)
                other_keys = {r.canonical_json(): r for _, r, _ in reads[other].rows}
                only_base = base_set - other_set
                only_other = other_set - base_set
                if len(only_base) == 0 and len(only_other) == 0:
                    continue
                examples = sorted(
                    {
                        canonical_json((base_keys.get(j) or other_keys.get(j)).key)
                        for j in list(only_base) + list(only_other)
                    }
                )[:_EXAMPLES]
                problems.append(
                    _problem(
                        "replicated-divergence",
                        name,
                        f"shard #{other} differs from shard #{base} in "
                        f"{sum(only_base.values()) + sum(only_other.values())} record(s) "
                        f"({sum(only_base.values())} only on shard #{base}, "
                        f"{sum(only_other.values())} only on shard #{other}); e.g. keys "
                        f"{'; '.join(examples)}",
                    )
                )
    else:
        contributing = sorted(reads)

    rows = [
        (s, serial, record, ts)
        for s in contributing
        for serial, record, ts in reads[s].rows
    ]

    by_identity: Dict[str, List[Tuple[int, int]]] = {}
    for s, serial, record, _ in rows:
        by_identity.setdefault(record.identity(), []).append((s, serial))
    duplicated = {i: where for i, where in by_identity.items() if len(where) > 1}
    if len(duplicated) > 0:
        examples = [
            f"shard/serial {', '.join(f'#{s}/{serial}' for s, serial in sorted(where))}"
            for _, where in sorted(duplicated.items())[:_EXAMPLES]
        ]
        problems.append(
            _problem(
                "duplicate",
                name,
                f"{len(duplicated)} key-and-tag set(s) are held by more than one record "
                f"({sum(len(w) for w in duplicated.values())} records, all kept); e.g. "
                f"{'; '.join(examples)}",
            )
        )

    stamps = [ts for _, _, _, ts in rows if ts is not None]
    records = tuple(sorted((r for _, _, r, _ in rows), key=Record.canonical_json))
    return ClassInventory(
        name=name,
        replicated=replicated,
        tagged=first.tagged,
        parents=dict(first.parents),
        records=records,
        count=len(records),
        earliest_timestamp=min(stamps) if len(stamps) > 0 else None,
        latest_timestamp=max(stamps) if len(stamps) > 0 else None,
        problems=tuple(problems),
    )


def read_inventory(primary: PathType) -> StoreInventory:
    """
    The structured inventory of the closed store whose primary is ``primary``.

    Read through ``open_read_only``: every file ``mode=ro``, a journal refused, no Ray. Each class
    in ``INVENTORY_CLASSES`` is built in that order by its factory's ``inventory_records`` on every
    shard, and then combined: a sharded class as the union, a replicated class compared across
    shards (module docstring).
    """
    from Datastore.SQL.Datastore import _factories
    from Datastore.store_reader import open_read_only
    from config.sharding import replicated_tables

    replicated = set(replicated_tables)

    with open_read_only(primary) as store, contextlib.ExitStack() as stack:
        conns = {
            shard.serial: stack.enter_context(shard.engine.connect())
            for shard in store.shards
        }
        keys: Dict[int, Dict[str, Dict[int, Mapping[str, Any]]]] = {
            shard.serial: {} for shard in store.shards
        }
        digests: Dict[int, Dict[str, Dict[int, str]]] = {
            shard.serial: {} for shard in store.shards
        }

        classes: Dict[str, ClassInventory] = {}
        for name in INVENTORY_CLASSES:
            factory = _factories[name]
            reads: Dict[int, ShardRead] = {}
            for shard in store.shards:
                context = ShardContext(
                    shard=shard.serial,
                    absent_tables=shard.absent_tables,
                    absent_columns=shard.absent_columns,
                    keys=keys[shard.serial],
                    digests=digests[shard.serial],
                )
                read = factory.inventory_records(
                    conns[shard.serial], shard.tables[name], shard.tables, context
                )
                reads[shard.serial] = read
                keys[shard.serial][name] = {s: r.key for s, r, _ in read.rows}
                digests[shard.serial][name] = {
                    s: reference_digest(r.key, r.tags, read.tagged)
                    for s, r, _ in read.rows
                }
            classes[name] = _combine(name, name in replicated, reads)

        return StoreInventory(
            primary=store.primary,
            shards=tuple(shard.serial for shard in store.shards),
            classes=classes,
        )
