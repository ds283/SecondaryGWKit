"""
A structured inventory of a **closed** ShardedPool store: what it holds, named by physical labels.

``read_inventory(primary, factories)`` opens the store with ``open_read_only`` (prompt 01) and
returns a ``StoreInventory``. ``factories`` is the registry of storable classes, which the caller
gives (prompts/datastore-generic, prompt 06). For each class it holds ``records``, their
``count``, the earliest and latest ``timestamp`` (for display only: no timestamp is in any record)
and ``problems``, a list of named strings. It reads; it never writes, and it needs no Ray.

**A record** (``Record``) is a small JSON-safe object:

- ``key``: the class's physical identity. Its leaves are canonical (``canonical``), and a reference
  to a parent row is the **digest of that parent's own canonical identity** (``reference_digest``),
  never its serial. The parent itself is a record of its own class. No serial or timestamp is in a
  key; a label or name is in one only where the class declares it a leaf, because it *is* the
  identity;
- ``tags``: the sorted tuple of the tag labels on the row's association rows, ``()`` where the
  class has no association table. The tag table and its columns are the layer's own
  (``Datastore.contract``);
- ``validated``: the row's flag, or ``None`` where the class has no such column;
- ``value_count``: the number of rows in the class's value table whose parent is this row, or
  ``None`` where the class has none. The value tables are counted, one ``GROUP BY`` per table per
  shard, and never listed.

A computed-values field can be added to ``Record`` later, beside these four and outside ``key``,
without changing what they are.

**Who defines identity.** Each factory declares its key, as data, in a static ``inventory_spec()``
beside its existing ``inventory()``: an ``InventorySpec`` naming the leaf columns, the parent
references and the association and value tables (prompts/datastore-generic, prompt 08). A factory
whose class is not in the inventory declares none (``SQLAFactoryBase.inventory_spec`` returns
``None``). This module does the reading, in ``read_records``.

**Which classes, in which order.** The classes are those of the registry that declare a spec, and
their order is derived from what the specs declare (``inventory_classes``): first the classes that
reference nothing, in registry order; then, one at a time, the earliest class in registry order
whose parents have all been placed. A class depends on every class its parents and parent sets
name, every class a polymorphic parent's type map names, and the tag table if it is tagged. The
driver builds classes in that order and hands ``read_records``, in ``context``, a map from serial
to canonical key and to reference digest for every class already built **on that shard**, and a
map from serial to reference digest for every class already built on **every** shard, which a
``cross_shard`` parent resolves against (the store is still read once, and only for reading).

**A polymorphic parent.** A key field may reference a row whose class is named per row by a type
column: ``Parent(column, type_column=..., types={type value: class, ...})``. The referencing
factory declares the map; the layer knows only that one column names the class and another holds
the serial.

**Floats (decision D1).** ``canonical`` is the one function that turns a leaf into its canonical
form. A float becomes ``float.hex`` of the value **as stored**; an integer, string, boolean or
``None`` is kept as it is. The lookups match floats within 1e-7; the key records the stored bits.

**Shards.** A sharded class is the union of every shard's records. A replicated class is read from
every shard and compared on key, tags, validated flag and value count; its records are the
lowest-serial shard's, and any shard that differs is a named problem (``replicated-divergence``).
Which classes are replicated is what the store's primary records in its ``replicated_tables`` table
(``ReadOnlyStore.replicated_tables``), not anything this module or its caller says. Nothing is
repaired.

**Named problems.** Each problem string starts with its name:

- ``replicated-divergence``: a replicated class differs between shards;
- ``duplicate``: two or more records of a class share a key and a tag set. All are kept;
- ``orphan-value``: value rows whose parent row is not on their shard;
- ``orphan-tag``: association rows whose parent row or whose tag is not on their shard;
- ``unresolved-parent``: rows referencing a parent row that cannot be resolved. They are not
  records;
- ``empty-parent-set``: rows of a class keyed on a set of parents (``ParentSet``) that have no
  member rows. They are not records;
- ``orphan-member``: member rows of a ``ParentSet`` table whose owning row is not on their shard.

**A set of parents** (``ParentSet``, prompts/datastore-integrity prompt 09b). A row may be
assembled from several parent rows, recorded in a member table with one member row per parent
combination, each naming the row that owns it. Such a key field's value is the digest of its
member fields' names and the sorted list of its resolved members, each member the tuple, in those
fields' order, of the reference digests of the rows it names (``None`` for a nullable field that
names none). Like every parent reference it is formed from the parents' own identities, never
their serials; the members are kept beside the records, in ``ClassInventory.parent_sets``, so that
a display can render the set by its physical labels. The member table is read once per shard,
read-only, as the tag and value tables are.

This module imports only the standard library, ``sqlalchemy`` and ``Datastore.contract`` at module
scope, so a factory can import it from inside its ``inventory_spec``. The reader is imported inside
``read_inventory``, and the factory registry is given to it.
"""

import contextlib
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import sqlalchemy as sqla

from Datastore.contract import TAG_LABEL, TAG_SERIAL, TAG_TABLE

PathType = Union[str, os.PathLike]

# up to this many examples are named in a problem
_EXAMPLES = 5


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

    The parent's class is named in one of two ways, and exactly one is given:

    - ``of``: the parent class;
    - ``types``, with ``type_column``: a **polymorphic** reference, whose class is named per row by
      the value in ``type_column``, through ``types`` (type value -> class). The referencing
      factory declares the map. A type value the map does not hold is a parent that cannot be
      resolved. ``type_column`` is read with the reference, and must also be one of the class's
      leaves, so that a key carries the type its reference was resolved through.

    ``nullable``: whether a NULL in ``column`` is a value of the key rather than a missing parent.
    A NULL there gives the key field ``None`` (prompts/datastore-integrity prompt 07: a row
    computed without the optional parent). On a parent that is not nullable a NULL is what it
    always was, a parent that cannot be resolved.

    ``cross_shard``: the parent row may be on another shard than the child's
    (prompts/datastore-integrity prompt 09a). The serial is then resolved against the parent
    class's records on **every** shard (``ShardContext.all_digests``), not this shard's alone.
    The resolved identity is the parent's own reference digest, as for a same-shard parent, never
    its serial. A serial that no shard holds, or that two shards hold with different records, is
    an unresolved parent."""

    column: str
    of: Optional[str] = None
    type_column: Optional[str] = None
    # not hashed, so that a Parent stays hashable; it is compared
    types: Optional[Mapping[Any, str]] = field(default=None, hash=False)
    nullable: bool = False
    cross_shard: bool = False

    def __post_init__(self):
        if (self.of is None) == (self.types is None):
            raise ValueError(
                f"Parent({self.column!r}): give exactly one of 'of' (the parent class) and "
                f"'types' (a polymorphic reference's type map)"
            )
        if (self.types is None) != (self.type_column is None):
            raise ValueError(
                f"Parent({self.column!r}): 'type_column' and 'types' are given together or not "
                f"at all"
            )
        if self.types is not None and len(self.types) == 0:
            raise ValueError(f"Parent({self.column!r}): 'types' is empty")

    def classes(self) -> Tuple[str, ...]:
        """Every class this reference can name, each once, in declaration order."""
        if self.types is None:
            return (self.of,)
        return tuple(dict.fromkeys(self.types.values()))

    def class_of(self, mapping: Mapping[str, Any]) -> Optional[str]:
        """The class this reference names in a row (or a key) ``mapping``: ``of``, or the class
        its type map gives the row's ``type_column``; ``None`` for a type the map does not hold.
        """
        if self.types is None:
            return self.of
        return self.types.get(mapping[self.type_column])


@dataclass(frozen=True)
class ParentSet:
    """A key field whose value is a **set** of parent references, one member per row of
    ``table`` whose ``owner`` column holds the child's serial (prompts/datastore-integrity prompt
    09b).

    ``members`` maps each field of a member to a ``Parent``, resolved as a key parent is, within
    this shard or, for a ``cross_shard`` one, on every shard. A member names its class (``of``):
    a polymorphic member is refused. A member naming a row that cannot be resolved makes the
    child an unresolved parent; a child with no member rows is an ``empty-parent-set``; a member
    row whose owner is not on the shard is an ``orphan-member``.
    """

    table: str
    owner: str
    members: Mapping[str, Parent]

    def __post_init__(self):
        polymorphic = [m for m, p in self.members.items() if p.types is not None]
        if len(polymorphic) > 0:
            raise ValueError(
                f"ParentSet({self.table!r}): member field(s) {', '.join(polymorphic)} are "
                f"polymorphic; a member of a parent set names its class ('of')"
            )


@dataclass(frozen=True)
class InventorySpec:
    """What a factory's ``inventory_spec()`` declares: how its class's records are read. The
    fields are ``read_records``' keywords, and mean what they mean there; ``keywords()`` gives
    them as such. A spec holds no connection and reads nothing."""

    leaves: Sequence[str] = ()
    parents: Mapping[str, Parent] = field(default_factory=dict)
    tags: Optional[Tuple[str, str]] = None
    values: Optional[Tuple[str, str]] = None
    validated: Optional[str] = None
    parent_sets: Mapping[str, ParentSet] = field(default_factory=dict)

    def keywords(self) -> Dict[str, Any]:
        """The spec as ``read_records``' keyword arguments."""
        return {
            "leaves": self.leaves,
            "parents": self.parents,
            "tags": self.tags,
            "values": self.values,
            "validated": self.validated,
            "parent_sets": self.parent_sets,
        }

    def dependencies(self) -> Tuple[str, ...]:
        """Every class that must be built before this one, each once: the classes its parents
        and its parent sets' members name (every class of a polymorphic parent's type map), and
        the tag table where the class is tagged, since its tags are read through the tag table's
        records."""
        named: List[str] = []
        for parent in self.parents.values():
            named.extend(parent.classes())
        for spec in self.parent_sets.values():
            for parent in spec.members.values():
                named.extend(parent.classes())
        if self.tags is not None:
            named.append(TAG_TABLE)
        return tuple(dict.fromkeys(named))


@dataclass(frozen=True)
class ShardContext:
    """What the driver hands ``read_records``, for one shard.

    ``keys[cls][serial]`` is the canonical key, and ``digests[cls][serial]`` the reference digest,
    of every row of every class already built on this shard. ``all_digests[cls][serial]`` is the
    reference digest of every row of every class already built on **every** shard, for a parent
    that may be on another one (``Parent.cross_shard``). A serial that two shards hold with
    different digests is absent from it, so that it resolves to nothing rather than to one of
    them."""

    shard: int
    keys: Mapping[str, Mapping[int, Mapping[str, Any]]]
    digests: Mapping[str, Mapping[int, str]]
    all_digests: Mapping[str, Mapping[int, str]] = field(default_factory=dict)


@dataclass
class ShardRead:
    """One class read from one shard by ``read_records``."""

    name: str
    shard: int
    rows: List[Tuple[int, Record, Optional[datetime]]]
    problems: List[str]
    tagged: bool
    # key field -> the parent class, or None for a polymorphic parent (parent_types)
    parents: Dict[str, Optional[str]]
    # key field -> member field -> the class that member field names (a ParentSet field)
    parent_sets: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # key field -> set digest -> the resolved members that digest is formed from, each a tuple in
    # the order of parent_sets[key field]
    members: Dict[str, Dict[str, Tuple[Tuple[Optional[str], ...], ...]]] = field(
        default_factory=dict
    )
    # key field -> (type column, type value -> class), for a polymorphic parent
    parent_types: Dict[str, Tuple[str, Mapping[Any, str]]] = field(default_factory=dict)


@dataclass(frozen=True)
class ClassInventory:
    """One class of the inventory. ``parents`` maps each key field that references a parent to
    the parent class, or to ``None`` for a polymorphic parent, whose type column and type map
    ``parent_types`` holds; ``parent_class`` names the class either way. ``tagged`` says whether
    the class has an association table, and so whether its reference digest covers tags.

    ``parent_sets`` maps each key field that is a set of parents (``ParentSet``) to its member
    fields and the class each names, and ``members`` maps that field and a set's digest, which is
    the field's value in a key, to the resolved members the digest is formed from, each a tuple of
    reference digests (or ``None``) in the order of ``parent_sets[field]``: a tuple, not a mapping,
    because a production store holds millions of members. Neither is in a record, so neither is in
    a fingerprint except through the digest."""

    name: str
    replicated: bool
    tagged: bool
    parents: Mapping[str, Optional[str]]
    records: Tuple[Record, ...]
    count: int
    earliest_timestamp: Optional[datetime]
    latest_timestamp: Optional[datetime]
    problems: Tuple[str, ...]
    parent_sets: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    members: Mapping[str, Mapping[str, Tuple[Tuple[Optional[str], ...], ...]]] = field(
        default_factory=dict
    )
    parent_types: Mapping[str, Tuple[str, Mapping[Any, str]]] = field(
        default_factory=dict
    )

    def parent_class(self, field: str, key: Mapping[str, Any]) -> Optional[str]:
        """The class that key field ``field`` of the record whose key is ``key`` references: the
        declared class, or for a polymorphic parent the class its type map gives the key's type
        column (``None`` for a type the map does not hold, or a field that is not a parent).
        """
        if field in self.parent_types:
            type_column, types = self.parent_types[field]
            return types.get(key.get(type_column))
        return self.parents.get(field)


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
        for field, value in record.key.items():
            if field in cls.parent_sets:
                out[field] = {
                    "set": [
                        {
                            member_field: self._resolve_reference(of, reference)
                            for (member_field, of), reference in zip(
                                cls.parent_sets[field].items(), member
                            )
                        }
                        for member in cls.members.get(field, {}).get(value, ())
                    ]
                }
                continue
            if field not in cls.parents:
                out[field] = value
                continue
            of = cls.parent_class(field, record.key)
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

    def _resolve_reference(self, of: str, value: Optional[str]) -> Any:
        """One member field of a parent set, resolved as ``resolve`` resolves a key parent."""
        if value is None:
            return None
        found = self.find(of, value)
        if len(found) != 1:
            return {"class": of, "digest": value, "unresolved": len(found)}
        return {
            "class": of,
            "key": self.resolve(of, found[0]),
            "tags": list(found[0].tags),
        }


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
    validated: Optional[str] = None,
    parent_sets: Optional[Mapping[str, ParentSet]] = None,
) -> ShardRead:
    """
    Read one class's records from one shard, as its factory's ``inventory_spec`` declares them
    (``InventorySpec.keywords()``).

    - ``leaves``: the key's leaf columns, each a key field of the same name, through ``canonical``;
    - ``parents``: key field -> ``Parent``, each resolved to the parent's reference digest through
      ``context``: on this shard, or on every shard for a ``cross_shard`` parent;
    - ``parent_sets``: key field -> ``ParentSet``, a set of parents recorded in a member table,
      each member resolved as a parent is, the field's value the digest of the sorted resolved
      members;
    - ``tags``: ``(association table, its column naming this class's serial)``, or ``None``;
    - ``values``: ``(value table, its column naming this class's serial)``, or ``None``;
    - ``validated``: the name of the class's validated-flag column, or ``None`` where it has
      none. It is read into the record's ``validated``.
    """
    name = table.name
    shard = context.shard
    parents = dict(parents or {})
    parent_sets = dict(parent_sets or {})
    problems: List[str] = []
    parent_classes = {f: p.of for f, p in parents.items()}
    parent_types = {
        f: (p.type_column, dict(p.types))
        for f, p in parents.items()
        if p.types is not None
    }
    set_classes = {
        f: {m: p.of for m, p in spec.members.items()} for f, spec in parent_sets.items()
    }
    set_members: Dict[str, Dict[str, Tuple[Tuple[Optional[str], ...], ...]]] = {
        f: {} for f in parent_sets
    }

    declared = list(parents.items()) + [
        (f"{f}.{m}", p)
        for f, spec in parent_sets.items()
        for m, p in spec.members.items()
    ]
    for field, parent in declared:
        for of in parent.classes():
            built = context.all_digests if parent.cross_shard else context.digests
            if of not in built:
                raise RuntimeError(
                    f"read_records(): {name}.{field} references {of}, which has not been built "
                    f"on shard #{shard}; classes must be built in dependency order"
                )

    def _read(rows):
        return ShardRead(
            name=name,
            shard=shard,
            rows=rows,
            problems=problems,
            tagged=tags is not None,
            parents=parent_classes,
            parent_sets=set_classes,
            members=set_members,
            parent_types=parent_types,
        )

    needed = list(
        dict.fromkeys(
            list(leaves)
            + [p.column for p in parents.values()]
            + [p.type_column for p in parents.values() if p.type_column is not None]
        )
    )

    columns = [table.c.serial] + [table.c[c] for c in needed]
    has_validated = validated is not None
    if has_validated:
        columns.append(table.c[validated].label("validated"))
    has_timestamp = "timestamp" in table.c
    if has_timestamp:
        columns.append(table.c.timestamp)

    rows = conn.execute(sqla.select(*columns).order_by(table.c.serial)).all()
    serials = {row.serial for row in rows}

    # the full tag set of each row, from its own association table, joined inside the shard
    tag_sets: Dict[int, List[str]] = {}
    if tags is not None:
        tag_table_name, parent_column = tags
        tag_table = tables[tag_table_name]
        labels = {
            serial: key[TAG_LABEL]
            for serial, key in context.keys.get(TAG_TABLE, {}).items()
        }
        no_parent: List[int] = []
        no_tag: List[int] = []
        for parent_serial, tag_serial in conn.execute(
            sqla.select(tag_table.c[parent_column], tag_table.c[TAG_SERIAL])
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
                    f"{TAG_TABLE} that is not there; e.g. tag serials {_examples(no_tag)}",
                )
            )

    # the number of value rows under each parent: one GROUP BY, never a listing
    counts: Dict[int, int] = {}
    if values is not None:
        value_table_name, parent_column = values
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

    # the member rows of each set of parents, grouped by the row that owns them: one read of each
    # member table, never one per row
    grouped: Dict[str, Dict[int, List[Any]]] = {}
    for field, spec in parent_sets.items():
        member_table = tables[spec.table]
        member_columns = list(
            dict.fromkeys([spec.owner] + [p.column for p in spec.members.values()])
        )
        by_owner: Dict[int, List[Any]] = {}
        orphan_members: Dict[int, int] = {}
        for member in conn.execute(
            sqla.select(*[member_table.c[c] for c in member_columns])
        ):
            owner = member._mapping[spec.owner]
            if owner in serials:
                by_owner.setdefault(owner, []).append(member)
            else:
                orphan_members[owner] = orphan_members.get(owner, 0) + 1
        if len(orphan_members) > 0:
            problems.append(
                _problem(
                    "orphan-member",
                    name,
                    f"{sum(orphan_members.values())} {spec.table} row(s) on shard #{shard} name "
                    f"{len(orphan_members)} {name} row(s) that are not there; e.g. owner "
                    f"serials {_examples(orphan_members)}",
                )
            )
        grouped[field] = by_owner

    def _reference(parent: Parent, mapping) -> Tuple[bool, Optional[str]]:
        """(resolved, the reference digest) of one parent reference in ``mapping``."""
        if parent.nullable and mapping[parent.column] is None:
            return True, None
        of = parent.class_of(mapping)
        # a cross-shard parent resolves against every shard's records of its class
        resolving = context.all_digests if parent.cross_shard else context.digests
        reference = (
            resolving.get(of, {}).get(mapping[parent.column])
            if of is not None
            else None
        )
        return reference is not None, reference

    out: List[Tuple[int, Record, Optional[datetime]]] = []
    unresolved: List[Tuple[int, str]] = []
    empty: List[Tuple[int, str]] = []
    for row in rows:
        mapping = row._mapping
        key: Dict[str, Any] = {leaf: canonical(mapping[leaf]) for leaf in leaves}
        missing_parent = None
        for field, parent in parents.items():
            resolved, reference = _reference(parent, mapping)
            if not resolved:
                missing_parent = field
                break
            key[field] = reference
        if missing_parent is not None:
            unresolved.append((row.serial, missing_parent))
            continue

        empty_set = None
        for field, spec in parent_sets.items():
            members = grouped[field].get(row.serial, [])
            if len(members) == 0:
                empty_set = field
                break
            resolved_members = []
            for member in members:
                resolved_member = []
                for parent in spec.members.values():
                    resolved, reference = _reference(parent, member._mapping)
                    if not resolved:
                        missing_parent = field
                        break
                    resolved_member.append(reference)
                if missing_parent is not None:
                    break
                resolved_members.append(tuple(resolved_member))
            if missing_parent is not None:
                break
            resolved_members.sort(
                key=lambda m: tuple("" if r is None else r for r in m)
            )
            key[field] = digest(
                {
                    "fields": list(spec.members),
                    "members": [list(m) for m in resolved_members],
                }
            )
            set_members[field][key[field]] = tuple(resolved_members)
        if missing_parent is not None:
            unresolved.append((row.serial, missing_parent))
            continue
        if empty_set is not None:
            empty.append((row.serial, empty_set))
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

    if len(empty) > 0:
        fields = sorted({f for _, f in empty})
        problems.append(
            _problem(
                "empty-parent-set",
                name,
                f"{len(empty)} row(s) on shard #{shard} have no member rows for their set of "
                f"parents (field(s) {', '.join(fields)}), and are not records; e.g. serials "
                f"{_examples(s for s, _ in empty)}",
            )
        )

    return _read(out)


# ---------------------------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------------------------


def _merge_digests(per_shard: Sequence[Mapping[int, str]]) -> Dict[int, str]:
    """One class's reference digests across shards, by serial. Serials are unique across shards
    for a sharded class, and equal on every shard for a replicated one. A serial two shards hold
    with different digests is left out: a cross-shard parent must resolve to one row or to none.
    """
    merged: Dict[int, str] = {}
    ambiguous = set()
    for digests in per_shard:
        for serial, reference in digests.items():
            if serial in merged and merged[serial] != reference:
                ambiguous.add(serial)
            merged.setdefault(serial, reference)
    return {s: d for s, d in merged.items() if s not in ambiguous}


def _combine(
    name: str, replicated: bool, reads: Mapping[int, ShardRead]
) -> ClassInventory:
    problems: List[str] = [p for s in sorted(reads) for p in reads[s].problems]
    first = reads[min(reads)]

    if replicated:
        # every shard is compared with the first: a shard whose files differ from the declared
        # tables was refused before anything was read
        shards = sorted(reads)
        contributing = shards[:1]
        base = shards[0]
        base_set = Counter(r.canonical_json() for _, r, _ in reads[base].rows)
        base_keys = {r.canonical_json(): r for _, r, _ in reads[base].rows}
        for other in shards[1:]:
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
    members: Dict[str, Dict[str, Tuple[Tuple[Optional[str], ...], ...]]] = {
        f: {} for f in first.parent_sets
    }
    for s in contributing:
        for f, sets in reads[s].members.items():
            members.setdefault(f, {}).update(sets)
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
        parent_sets={f: dict(m) for f, m in first.parent_sets.items()},
        members=members,
        parent_types=dict(first.parent_types),
    )


def inventory_specs(factories: Mapping[str, Any]) -> Dict[str, InventorySpec]:
    """Each class of the registry whose factory declares an ``inventory_spec``, mapped to it, in
    registry order. A factory with no such hook, or whose hook returns ``None``, is not a class of
    the inventory."""
    specs: Dict[str, InventorySpec] = {}
    for name, factory in factories.items():
        hook = getattr(factory, "inventory_spec", None)
        spec = hook() if hook is not None else None
        if spec is not None:
            specs[name] = spec
    return specs


def inventory_classes(factories: Mapping[str, Any]) -> List[str]:
    """
    The classes of the inventory, in the order they are built: the classes of the registry that
    declare an ``inventory_spec``, ordered from what the specs declare.

    First the **roots**, the classes that depend on nothing (``InventorySpec.dependencies``), in
    registry order; then, one at a time, the earliest remaining class in registry order whose
    dependencies have all been placed. So every class comes after every class it references.

    A spec whose dependency is not a class of the inventory, a polymorphic parent whose type
    column is not one of its class's leaves, or a cycle, raises ``ValueError`` naming the classes.
    """
    specs = inventory_specs(factories)
    depends: Dict[str, Tuple[str, ...]] = {}
    for name, spec in specs.items():
        missing = [d for d in spec.dependencies() if d not in specs]
        if len(missing) > 0:
            raise ValueError(
                f"inventory_classes(): {name} references {', '.join(missing)}, which "
                f"declare{'s' if len(missing) == 1 else ''} no inventory_spec"
            )
        for field_name, parent in spec.parents.items():
            if parent.types is not None and parent.type_column not in spec.leaves:
                raise ValueError(
                    f"inventory_classes(): {name}.{field_name} is a polymorphic parent whose "
                    f"type column {parent.type_column} is not one of {name}'s leaves"
                )
        depends[name] = tuple(d for d in spec.dependencies() if d != name)
        if name in spec.dependencies():
            raise ValueError(f"inventory_classes(): {name} references itself")

    order: List[str] = [name for name in specs if len(depends[name]) == 0]
    placed = set(order)
    while len(order) < len(specs):
        ready = [
            name
            for name in specs
            if name not in placed and all(d in placed for d in depends[name])
        ]
        if len(ready) == 0:
            stuck = [name for name in specs if name not in placed]
            raise ValueError(
                f"inventory_classes(): the references of {', '.join(stuck)} form a cycle, "
                f"or depend on one; no order builds every parent before its child"
            )
        order.append(ready[0])
        placed.add(ready[0])
    return order


def read_inventory(primary: PathType, factories: Mapping[str, Any]) -> StoreInventory:
    """
    The structured inventory of the closed store whose primary is ``primary``.

    Read through ``open_read_only``: every file ``mode=ro``, a journal refused, no Ray. Each class
    of ``inventory_classes(factories)`` is read in that order, as its factory's ``inventory_spec``
    declares, on every shard, and then combined: a sharded class as the union, a replicated class
    compared across shards (module docstring). ``factories`` is the registry of storable classes;
    which classes are replicated is read from the store's primary.
    """
    from Datastore.store_reader import open_read_only

    with open_read_only(primary, factories) as store, contextlib.ExitStack() as stack:
        # derived inside the block, after the reader has checked the registry against the store: a
        # registry that does not fit its store is refused by the reader, as it always was, and not
        # by a refusal of the order
        names = inventory_classes(factories)
        specs = inventory_specs(factories)
        replicated = set(store.replicated_tables)
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

        # every class already built, on every shard, for a parent that may be on another shard
        all_digests: Dict[str, Dict[int, str]] = {}

        classes: Dict[str, ClassInventory] = {}
        for name in names:
            keywords = specs[name].keywords()
            reads: Dict[int, ShardRead] = {}
            for shard in store.shards:
                context = ShardContext(
                    shard=shard.serial,
                    keys=keys[shard.serial],
                    digests=digests[shard.serial],
                    all_digests=all_digests,
                )
                read = read_records(
                    conns[shard.serial],
                    shard.tables[name],
                    shard.tables,
                    context,
                    **keywords,
                )
                reads[shard.serial] = read
                keys[shard.serial][name] = {s: r.key for s, r, _ in read.rows}
                digests[shard.serial][name] = {
                    s: reference_digest(r.key, r.tags, read.tagged)
                    for s, r, _ in read.rows
                }
            all_digests[name] = _merge_digests(
                [digests[shard.serial][name] for shard in store.shards]
            )
            classes[name] = _combine(name, name in replicated, reads)

        return StoreInventory(
            primary=store.primary,
            shards=tuple(shard.serial for shard in store.shards),
            classes=classes,
        )
