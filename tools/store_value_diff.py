"""
Compare the **values** of two closed stores, row by row, matched by physical identity
(``prompts/levin-phase-floor`` prompt 05 §3A; ``CLAUDE.md``, campaign conventions rule 7).

    PYTHONPATH=. ./venv/bin/python -m tools.store_value_diff A.sqlite B.sqlite [--json]
        [--classes NAME ...] [--max-rows N]

**Why.** A store fingerprint (``datastorekit.store_fingerprint``) certifies what a store holds --
identities, tags, value-row counts -- and never a computed value. A change to how values are
computed is certified by a rehearsal pair, one before and one after, compared by this tool: every
value outside the classes the change is meant to touch must be bit-identical, and the differences
inside them are what it reports.

**Identity, never serials.** Each store is read as ``datastorekit.store_inventory.read_inventory``
reads it -- ``open_read_only`` (every file ``mode=ro``, no Ray), the classes in
``inventory_classes`` order, each through its factory's ``inventory_spec`` with ``read_records`` --
and the serial of every row is mapped to the identity the inventory gives it: its record's key and
tag set (``Record.identity``), in which every reference to another row is that row's own reference
digest. Two stores' rows are matched by that identity alone. A serial is never compared and never
used to match. This is the inventory's public reading, driven here so that each row's serial can
be kept beside its identity; nothing in ``datastorekit`` is changed or patched.

**What is compared, per class** (every class of the inventory, or those named by ``--classes``):

- **the row's own columns**, all of them except ``serial``, ``timestamp`` and the columns that name
  another row. A column naming a row that is one of the class's key parents is already compared
  through the key. Any other column naming a row (a foreign key outside the key, such as
  ``z_min_serial`` or ``solver_serial``) is compared through the identity of the row it names: its
  reference digest, never its serial;
- **every row of each value table** of the class (``InventorySpec.values`` and ``extra_values``),
  matched by its parent's identity and by the identity of every other row it names (a value row's
  redshift, for example). Its other columns are compared as a row's are.

A replicated class, and a replicated value table, is compared on its lowest-serial shard, which is
the shard whose records the inventory reports for it (the inventory names any divergence between
shards as a problem; ``store fingerprint`` reports it).

**How values are compared.** Floats bit for bit, by ``float.hex``: ``NaN`` equals ``NaN``, ``+inf``
equals ``+inf``, ``0.0`` differs from ``-0.0``. A string that holds a JSON object or array in both
stores (the ``metadata`` columns) is compared as JSON: parsed, objects compared key by key whatever
their order, arrays index by index, leaves as above. Anything else by ``==``.

**What cannot be matched is said, by name, and never compared by serial instead.** A row is
unmatchable when it is not a record (a parent that does not resolve, an empty parent set), or when
two rows of its class in one store share an identity; a value row when its parent is not a record,
when a row it names does not resolve, or when two value rows in one store share a match key. A
column that looks like a reference (``*_serial``) but declares no target cannot be compared by
identity, and is named. A table that is neither a class of the inventory nor one of their value,
tag or member tables is named as not compared. Each makes the result "not value-identical".

**What it reports.** Per class: rows identical, rows that differ, rows only in one store, rows that
cannot be matched; per differing column the count and, for floats, the largest ``|Δ|`` and
``|Δ|/|a|`` (``a`` the first store's value); for a JSON column, the keys that differ; the same for
each value table. Then the differing rows, each by its identity (its reference digest and its key
resolved to physical values), with each differing column's two values and ``Δ``. Text for a person
by default (at most ``--max-rows`` rows per class); with ``--json``, the whole of it as JSON.

**A column one store's files do not hold** (``prompts/levin-phase-floor`` prompt 06 §1.5). The
code declares the tables this tool reads; a store written by an earlier commit can lack a column the
code has since added (06's ``theta_residual``), and ``open_read_only`` refuses such a store whole.
When that is the only way a store differs from the declared tables -- declared columns its files
lack, and nothing else -- it is read without those columns, and each is reported, per table, as
present in one store only. Every other column is compared as before. Any other schema difference
(a table absent or undeclared, a column the code does not declare) is still a refusal, exit 2. A
column present in one store only makes the result "not value-identical".

**Exit status.** 0 when the stores are value-identical in every class compared; 1 otherwise,
including when anything could not be matched. 2 for a usage error or a store that cannot be opened.
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import sqlalchemy as sqla

from datastorekit.SQL.schema import StoreSchemaMismatch
from datastorekit.contract import TAG_TABLE
from datastorekit.store_reader import open_read_only
from datastorekit.store_inventory import (
    ShardContext,
    canonical_json,
    inventory_classes,
    inventory_specs,
    read_records,
    reference_digest,
)

# the columns of a row that are never compared: its own serial and its write time
_NOT_COMPARED = ("serial", "timestamp")

# the default number of differing rows printed per class (all are in --json)
DEFAULT_MAX_ROWS = 20


# ---------------------------------------------------------------------------------------------
# reading one store
# ---------------------------------------------------------------------------------------------


@dataclass
class _Row:
    """One row or value row, by its match key: where it is (for a message only) and its values."""

    shard: int
    serial: int
    values: Dict[str, Any]


@dataclass
class _TableRead:
    """A class's rows, or one of its value tables' rows, read from one store."""

    # match key -> the rows that have it (more than one is a duplicate, and unmatchable)
    rows: Dict[Any, List[_Row]] = field(default_factory=dict)
    # rows that could not be given a match key, each with the reason
    unmatched: List[str] = field(default_factory=list)
    # the columns compared, as this store holds them (a column its files lack is not among them)
    columns: Tuple[str, ...] = ()


@dataclass
class _ClassRead:
    name: str
    replicated: bool
    rows: _TableRead
    # value table -> its rows, keyed by (parent identity, ((column, class, identity) of each row
    # it names, ...))
    values: Dict[str, _TableRead]
    # columns that look like references but cannot be resolved to an identity, by table
    unresolvable_columns: Dict[str, List[str]]
    # identity -> (reference digest, key) for each record, for display
    records: Dict[str, Tuple[str, Mapping[str, Any]]]


@dataclass
class _StoreRead:
    primary: Path
    classes: Dict[str, _ClassRead]
    # (class, reference digest) -> (key, tags), for resolving a key for display
    by_digest: Dict[Tuple[str, str], Tuple[Mapping[str, Any], Tuple[str, ...]]]
    # class -> its parent fields' classes (None for a polymorphic one) and type maps
    parents: Dict[str, Dict[str, Any]]
    # tables that no class accounts for, never compared
    not_compared: List[str]
    # class -> {leaf column -> True if it is a Float column}
    float_leaves: Dict[str, Dict[str, bool]]
    # table -> the declared columns this store's files do not hold, read without them (prompt 06)
    absent_columns: Dict[str, Tuple[str, ...]] = field(default_factory=dict)


def _reference_columns(
    table: sqla.Table, exclude: Set[str]
) -> Tuple[Dict[str, str], List[str]]:
    """
    ``(resolvable, unresolvable)``: the columns of ``table`` outside ``exclude`` that name another
    row. A foreign key names its target table; a column called ``*_serial`` with no foreign key
    names a row this tool cannot identify.
    """
    resolvable: Dict[str, str] = {}
    unresolvable: List[str] = []
    for column in table.columns:
        if column.name in exclude or column.name in _NOT_COMPARED:
            continue
        targets = sorted({fk.column.table.name for fk in column.foreign_keys})
        if len(targets) == 1:
            resolvable[column.name] = targets[0]
        elif len(targets) > 1 or column.name.endswith("_serial"):
            unresolvable.append(column.name)
    return resolvable, unresolvable


def _identity_of(
    target: str,
    serial: Optional[int],
    shard: int,
    digests: Mapping[int, Mapping[str, Mapping[int, str]]],
    all_digests: Mapping[str, Mapping[int, str]],
) -> Tuple[bool, Optional[str]]:
    """``(resolved, reference digest)`` of row ``serial`` of ``target``: on this shard first, then
    on every shard (a row a sharded class names may be on another shard). ``None`` stays ``None``.
    """
    if serial is None:
        return True, None
    found = digests.get(shard, {}).get(target, {}).get(serial)
    if found is None:
        found = all_digests.get(target, {}).get(serial)
    return found is not None, found


class _WithoutColumns:
    """
    A factory whose table omits the declared columns a store's files do not hold, so that
    ``open_read_only`` can read that store; every other attribute is the factory's own.
    """

    def __init__(self, factory, absent: Set[str]):
        self._factory = factory
        self._absent = set(absent)

    def register(self):
        declared = self._factory.register()
        if declared is None:
            return None
        out = dict(declared)
        out["columns"] = [c for c in declared["columns"] if c.name not in self._absent]
        return out

    def __getattr__(self, name):
        return getattr(self._factory, name)


def _without_columns(
    factories: Mapping[str, Any], absent: Mapping[str, Set[str]]
) -> Dict[str, Any]:
    return {
        name: (_WithoutColumns(f, absent[name]) if name in absent else f)
        for name, f in factories.items()
    }


def _absent_columns(primary, factories: Mapping[str, Any]) -> Dict[str, Set[str]]:
    """
    The declared columns ``primary``'s files lack, by table, when that is the only way they differ
    from the declared tables; ``{}`` when they hold exactly the declared tables. Any other
    difference is ``open_read_only``'s refusal, re-raised as it was raised. The store is opened
    read-only and closed again, once per table found short.
    """
    absent: Dict[str, Set[str]] = {}
    while True:
        try:
            with open_read_only(primary, _without_columns(factories, absent)):
                return absent
        except StoreSchemaMismatch as e:
            d = e.differences
            if (
                len(d.absent_tables) > 0
                or len(d.extra_tables) > 0
                or len(d.extra_columns) > 0
                or len(d.absent_columns) == 0
                or any(table not in factories for table in d.absent_columns)
            ):
                raise
            grew = False
            for table, columns in d.absent_columns.items():
                before = len(absent.get(table, ()))
                absent.setdefault(table, set()).update(columns)
                grew = grew or len(absent[table]) > before
            if not grew:
                raise


def read_store(
    primary, factories: Mapping[str, Any], classes: Optional[Sequence[str]] = None
) -> _StoreRead:
    """
    Read the closed store ``primary``, read-only and with no Ray, into rows matched by identity.
    Every class of the inventory is read for its identities (a child names its parents by them);
    the values of ``classes`` (all, by default) are kept for comparison.
    """
    # a store written before a column was added is read without it (module docstring)
    absent = _absent_columns(primary, factories)
    factories = _without_columns(factories, absent)

    names = inventory_classes(factories)
    specs = inventory_specs(factories)
    wanted = set(names if classes is None else classes)

    with open_read_only(primary, factories) as store:
        replicated = set(store.replicated_tables)
        shard_serials = [shard.serial for shard in store.shards]
        lowest = min(shard_serials)
        conns = {}
        try:
            for shard in store.shards:
                conns[shard.serial] = shard.engine.connect()

            keys: Dict[int, Dict[str, Dict[int, Mapping[str, Any]]]] = {
                s: {} for s in shard_serials
            }
            digests: Dict[int, Dict[str, Dict[int, str]]] = {
                s: {} for s in shard_serials
            }
            all_digests: Dict[str, Dict[int, str]] = {}
            # shard -> class -> serial -> identity (Record.identity), for every class
            identities: Dict[int, Dict[str, Dict[int, str]]] = {
                s: {} for s in shard_serials
            }
            by_digest: Dict[
                Tuple[str, str], Tuple[Mapping[str, Any], Tuple[str, ...]]
            ] = {}
            parents_of: Dict[str, Dict[str, Any]] = {}
            float_leaves: Dict[str, Dict[str, bool]] = {}
            out: Dict[str, _ClassRead] = {}

            # the inventory's reading, in its order (read_inventory's driver, with the serials kept)
            for name in names:
                spec = specs[name]
                keywords = spec.keywords()
                tagged = spec.tags is not None
                records_by_identity: Dict[str, Tuple[str, Mapping[str, Any]]] = {}
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
                    keys[shard.serial][name] = {s: r.key for s, r, _ in read.rows}
                    digests[shard.serial][name] = {
                        s: reference_digest(r.key, r.tags, tagged)
                        for s, r, _ in read.rows
                    }
                    identities[shard.serial][name] = {
                        s: r.identity() for s, r, _ in read.rows
                    }
                    for s, r, _ in read.rows:
                        d = reference_digest(r.key, r.tags, tagged)
                        by_digest[(name, d)] = (r.key, tuple(r.tags))
                        records_by_identity[r.identity()] = (d, r.key)
                merged: Dict[int, str] = {}
                ambiguous = set()
                for s in shard_serials:
                    for serial, d in digests[s][name].items():
                        if serial in merged and merged[serial] != d:
                            ambiguous.add(serial)
                        merged.setdefault(serial, d)
                all_digests[name] = {
                    s: d for s, d in merged.items() if s not in ambiguous
                }

                parents_of[name] = {
                    f: (p.of, p.type_column, dict(p.types) if p.types else None)
                    for f, p in spec.parents.items()
                }
                table = store.tables[name]
                float_leaves[name] = {
                    leaf: isinstance(table.c[leaf].type, sqla.Float)
                    for leaf in spec.leaves
                }

                if name not in wanted:
                    continue
                out[name] = _read_class(
                    name,
                    spec,
                    store,
                    conns,
                    replicated,
                    lowest,
                    identities,
                    digests,
                    all_digests,
                    records_by_identity,
                )

            not_compared = _tables_not_accounted_for(store.tables, specs)
        finally:
            for conn in conns.values():
                conn.close()

        return _StoreRead(
            primary=store.primary,
            classes=out,
            by_digest=by_digest,
            parents=parents_of,
            not_compared=not_compared,
            float_leaves=float_leaves,
            absent_columns={t: tuple(sorted(c)) for t, c in absent.items()},
        )


def _tables_not_accounted_for(tables: Mapping[str, sqla.Table], specs) -> List[str]:
    """Tables that are neither a class of the inventory nor one of its value, tag or member tables."""
    accounted = set(specs) | {TAG_TABLE}
    for spec in specs.values():
        if spec.values is not None:
            accounted.add(spec.values[0])
        accounted.update(spec.extra_values)
        if spec.tags is not None:
            accounted.add(spec.tags[0])
        for ps in spec.parent_sets.values():
            accounted.add(ps.table)
    return sorted(t for t in tables if t not in accounted)


def _read_class(
    name,
    spec,
    store,
    conns,
    replicated,
    lowest,
    identities,
    digests,
    all_digests,
    records_by_identity,
) -> _ClassRead:
    table = store.tables[name]
    shards = [s for s in store.shards if name not in replicated or s.serial == lowest]

    # the row's own columns: everything but serial, timestamp and the key's parent references;
    # any other column naming a row is compared through that row's identity
    parent_columns = {p.column for p in spec.parents.values()}
    refs, unresolvable = _reference_columns(table, parent_columns)
    unresolvable_columns: Dict[str, List[str]] = {}
    if len(unresolvable) > 0:
        unresolvable_columns[name] = unresolvable
    compared = [
        c.name
        for c in table.columns
        if c.name not in _NOT_COMPARED
        and c.name not in parent_columns
        and c.name not in unresolvable
    ]

    rows = _TableRead(columns=tuple(compared))
    for shard in shards:
        serial_identity = identities[shard.serial][name]
        for row in conns[shard.serial].execute(
            sqla.select(*[shard.tables[name].c[c] for c in ["serial"] + compared])
        ):
            mapping = row._mapping
            identity = serial_identity.get(mapping["serial"])
            if identity is None:
                rows.unmatched.append(
                    f"shard #{shard.serial} serial {mapping['serial']}: not a record of the "
                    f"inventory (a parent that does not resolve, or an empty parent set)"
                )
                continue
            values, problem = _resolve_values(
                mapping, compared, refs, shard.serial, digests, all_digests
            )
            if problem is not None:
                rows.unmatched.append(
                    f"shard #{shard.serial} serial {mapping['serial']}: {problem}"
                )
                continue
            rows.rows.setdefault(identity, []).append(
                _Row(shard.serial, mapping["serial"], values)
            )

    value_tables: Dict[str, str] = {}
    if spec.values is not None:
        value_tables[spec.values[0]] = spec.values[1]
    value_tables.update(spec.extra_values)

    values_read: Dict[str, _TableRead] = {}
    for value_table, parent_column in sorted(value_tables.items()):
        vt = store.tables[value_table]
        v_refs, v_unresolvable = _reference_columns(vt, {parent_column})
        if len(v_unresolvable) > 0:
            unresolvable_columns[value_table] = v_unresolvable
        ref_names = sorted(v_refs)
        v_compared = [
            c.name
            for c in vt.columns
            if c.name not in _NOT_COMPARED
            and c.name != parent_column
            and c.name not in v_refs
            and c.name not in v_unresolvable
        ]
        v_shards = [
            s
            for s in store.shards
            if value_table not in replicated or s.serial == lowest
        ]
        read = _TableRead(columns=tuple(v_compared))
        for shard in v_shards:
            serial_identity = identities[shard.serial][name]
            columns = ["serial", parent_column] + ref_names + v_compared
            for row in conns[shard.serial].execute(
                sqla.select(*[shard.tables[value_table].c[c] for c in columns])
            ):
                mapping = row._mapping
                parent = serial_identity.get(mapping[parent_column])
                where = f"shard #{shard.serial} serial {mapping['serial']}"
                if parent is None:
                    read.unmatched.append(
                        f"{where}: its {name} row (serial {mapping[parent_column]}) is not a "
                        f"record of the inventory on this shard"
                    )
                    continue
                named = []
                problem = None
                for column in ref_names:
                    ok, identity = _identity_of(
                        v_refs[column],
                        mapping[column],
                        shard.serial,
                        digests,
                        all_digests,
                    )
                    if not ok:
                        problem = (
                            f"{where}: {column} names {v_refs[column]} serial "
                            f"{mapping[column]}, which does not resolve to an identity"
                        )
                        break
                    named.append((column, v_refs[column], identity))
                if problem is not None:
                    read.unmatched.append(problem)
                    continue
                key = (parent, tuple(named))
                read.rows.setdefault(key, []).append(
                    _Row(
                        shard.serial,
                        mapping["serial"],
                        {c: mapping[c] for c in v_compared},
                    )
                )
        values_read[value_table] = read

    return _ClassRead(
        name=name,
        replicated=name in replicated,
        rows=rows,
        values=values_read,
        unresolvable_columns=unresolvable_columns,
        records=records_by_identity,
    )


def _resolve_values(mapping, compared, refs, shard, digests, all_digests):
    """A row's compared values, with each non-key reference replaced by the named row's identity."""
    values: Dict[str, Any] = {}
    for column in compared:
        value = mapping[column]
        if column in refs:
            ok, identity = _identity_of(
                refs[column], value, shard, digests, all_digests
            )
            if not ok:
                return None, (
                    f"{column} names {refs[column]} serial {value}, which does not resolve to "
                    f"an identity"
                )
            value = _Reference(refs[column], identity)
        values[column] = value
    return values, None


@dataclass(frozen=True)
class _Reference:
    """A non-key column naming a row, by the named row's identity (its reference digest)."""

    target: str
    digest: Optional[str]


# ---------------------------------------------------------------------------------------------
# comparing values
# ---------------------------------------------------------------------------------------------

_NOT_JSON = object()


def _as_json(value: Any) -> Any:
    """``value`` parsed, if it is a string holding a JSON object or array; else ``_NOT_JSON``."""
    if not isinstance(value, str):
        return _NOT_JSON
    text = value.strip()
    if not text or text[0] not in "{[":
        return _NOT_JSON
    try:
        parsed = json.loads(text)
    except ValueError:
        return _NOT_JSON
    return parsed if isinstance(parsed, (dict, list)) else _NOT_JSON


def _leaf_equal(a: Any, b: Any) -> bool:
    """Floats bit for bit (``float.hex``), with an integer compared as the float it equals; else
    ``==`` and the same kind (so that ``True`` and ``1`` differ)."""
    a_float = isinstance(a, float)
    b_float = isinstance(b, float)
    if a_float or b_float:
        numeric = (int, float)
        if not (
            isinstance(a, numeric)
            and isinstance(b, numeric)
            and not isinstance(a, bool)
            and not isinstance(b, bool)
        ):
            return False
        return float.hex(float(a)) == float.hex(float(b))
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    return a == b


def _json_differences(a: Any, b: Any, path: str = "") -> List[str]:
    """The keys (dotted paths, ``[i]`` for an array index) at which two JSON values differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        out: List[str] = []
        for key in sorted(set(a) | set(b)):
            sub = f"{path}.{key}" if path else str(key)
            if key not in a or key not in b:
                out.append(sub)
            else:
                out.extend(_json_differences(a[key], b[key], sub))
        return out
    if isinstance(a, list) and isinstance(b, list):
        out = []
        for i in range(max(len(a), len(b))):
            sub = f"{path}[{i}]"
            if i >= len(a) or i >= len(b):
                out.append(sub)
            else:
                out.extend(_json_differences(a[i], b[i], sub))
        return out
    if isinstance(a, (dict, list)) or isinstance(b, (dict, list)):
        return [path or "(top)"]
    return [] if _leaf_equal(a, b) else [path or "(top)"]


def _json_safe_float(x: float) -> Any:
    """A float for the JSON report: itself when finite, else its name."""
    if math.isfinite(x):
        return x
    return "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")


def compare_values(a: Any, b: Any) -> Optional[Dict[str, Any]]:
    """
    ``None`` when ``a`` and ``b`` are the same value under this tool's rules (module docstring);
    otherwise a description of the difference: ``kind`` (``float``, ``json``, ``reference`` or
    ``value``), and for a float ``a``, ``b`` (as ``float.hex`` and as numbers), ``delta = b - a``
    and ``rel = |delta|/|a|``; for JSON the ``keys`` that differ.
    """
    if isinstance(a, _Reference) or isinstance(b, _Reference):
        if a == b:
            return None
        return {
            "kind": "reference",
            "a": None if not isinstance(a, _Reference) else a.digest,
            "b": None if not isinstance(b, _Reference) else b.digest,
        }

    ja = _as_json(a)
    jb = _as_json(b)
    if ja is not _NOT_JSON and jb is not _NOT_JSON:
        keys = _json_differences(ja, jb)
        if len(keys) == 0:
            return None
        return {"kind": "json", "keys": keys}

    if _leaf_equal(a, b):
        return None

    if (
        isinstance(a, (int, float))
        and isinstance(b, (int, float))
        and not isinstance(a, bool)
        and not isinstance(b, bool)
        and (isinstance(a, float) or isinstance(b, float))
    ):
        fa = float(a)
        fb = float(b)
        if math.isfinite(fa) and math.isfinite(fb):
            delta = fb - fa
            rel = abs(delta) / abs(fa) if fa != 0.0 else math.inf
        else:
            delta = math.inf
            rel = math.inf
        return {
            "kind": "float",
            "a": float.hex(fa),
            "b": float.hex(fb),
            "a_value": _json_safe_float(fa),
            "b_value": _json_safe_float(fb),
            "delta": _json_safe_float(delta),
            "rel": _json_safe_float(rel),
        }

    return {"kind": "value", "a": _plain(a), "b": _plain(b)}


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return _json_safe_float(value) if isinstance(value, float) else value
    return str(value)


@dataclass
class _ColumnSummary:
    count: int = 0
    max_abs_delta: Optional[float] = None
    max_rel_delta: Optional[float] = None
    json_keys: Dict[str, int] = field(default_factory=dict)

    def add(self, diff: Dict[str, Any]) -> None:
        self.count += 1
        if diff["kind"] == "float":
            d = diff["delta"]
            r = diff["rel"]
            d = math.inf if isinstance(d, str) else abs(d)
            r = math.inf if isinstance(r, str) else r
            self.max_abs_delta = (
                d if self.max_abs_delta is None else max(self.max_abs_delta, d)
            )
            self.max_rel_delta = (
                r if self.max_rel_delta is None else max(self.max_rel_delta, r)
            )
        elif diff["kind"] == "json":
            for key in diff["keys"]:
                self.json_keys[key] = self.json_keys.get(key, 0) + 1

    def as_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"count": self.count}
        if self.max_abs_delta is not None:
            out["max_abs_delta"] = _json_safe_float(self.max_abs_delta)
            out["max_rel_delta"] = _json_safe_float(self.max_rel_delta)
        if len(self.json_keys) > 0:
            out["json_keys"] = dict(sorted(self.json_keys.items()))
        return out


def _compare_tables(a: _TableRead, b: _TableRead, describe) -> Dict[str, Any]:
    """Match two reads of one table by key and compare every matched pair."""
    duplicated = sorted(
        {k for k, v in a.rows.items() if len(v) > 1}
        | {k for k, v in b.rows.items() if len(v) > 1},
        key=repr,
    )
    dup = set(duplicated)
    keys_a = {k for k in a.rows if k not in dup}
    keys_b = {k for k in b.rows if k not in dup}

    # a column one store does not hold is reported as such, and not compared row by row
    columns_a = set(a.columns)
    columns_b = set(b.columns)
    common = sorted(columns_a & columns_b)

    identical = 0
    columns: Dict[str, _ColumnSummary] = {}
    differing: List[Dict[str, Any]] = []
    for key in sorted(keys_a & keys_b, key=repr):
        va = a.rows[key][0].values
        vb = b.rows[key][0].values
        diffs = {}
        for column in common:
            diff = compare_values(va.get(column), vb.get(column))
            if diff is not None:
                diffs[column] = diff
                columns.setdefault(column, _ColumnSummary()).add(diff)
        if len(diffs) == 0:
            identical += 1
        else:
            entry = describe(key)
            entry["columns"] = diffs
            differing.append(entry)

    unmatched = [f"A: {m}" for m in a.unmatched] + [f"B: {m}" for m in b.unmatched]
    for key in duplicated:
        unmatched.append(
            f"{len(a.rows.get(key, []))} row(s) in A and {len(b.rows.get(key, []))} in B "
            f"share one identity: {describe(key)['identity']}"
        )

    return {
        "identical": identical,
        "differ": len(differing),
        "only_in_a": [describe(k) for k in sorted(keys_a - keys_b, key=repr)],
        "only_in_b": [describe(k) for k in sorted(keys_b - keys_a, key=repr)],
        "unmatched": unmatched,
        "columns_only_in_a": sorted(columns_a - columns_b),
        "columns_only_in_b": sorted(columns_b - columns_a),
        "columns": {c: s.as_json() for c, s in sorted(columns.items())},
        "rows": differing,
    }


# ---------------------------------------------------------------------------------------------
# identities for display
# ---------------------------------------------------------------------------------------------


class _Describer:
    """Resolves a key to physical values, recursively and memoised, from either store."""

    def __init__(self, stores: Sequence[_StoreRead]):
        self._by_digest: Dict[
            Tuple[str, str], Tuple[Mapping[str, Any], Tuple[str, ...]]
        ] = {}
        self._parents: Dict[str, Dict[str, Any]] = {}
        self._floats: Dict[str, Dict[str, bool]] = {}
        for store in stores:
            self._by_digest.update(store.by_digest)
            self._parents.update(store.parents)
            self._floats.update(store.float_leaves)
        self._memo: Dict[Tuple[str, str], Any] = {}

    def key(self, name: str, key: Mapping[str, Any]) -> Dict[str, Any]:
        parents = self._parents.get(name, {})
        floats = self._floats.get(name, {})
        out: Dict[str, Any] = {}
        for f, value in key.items():
            if f in parents:
                of, type_column, types = parents[f]
                if types is not None:
                    of = types.get(key.get(type_column))
                out[f] = self.reference(of, value)
            elif floats.get(f) and isinstance(value, str):
                out[f] = _json_safe_float(float.fromhex(value))
            else:
                out[f] = value
        return out

    def reference(self, of: Optional[str], digest: Optional[str]) -> Any:
        if digest is None or of is None:
            return None
        memo_key = (of, digest)
        if memo_key not in self._memo:
            found = self._by_digest.get(memo_key)
            if found is None:
                self._memo[memo_key] = {"class": of, "digest": digest}
            else:
                key, tags = found
                resolved = {"class": of, **self.key(of, key)}
                if len(tags) > 0:
                    resolved["tags"] = list(tags)
                self._memo[memo_key] = resolved
        return self._memo[memo_key]


def _compact(value: Any, depth: int = 0, max_depth: int = 2) -> str:
    """One line for a person: a resolved key as ``Class(f=v, ...)``, nested to ``max_depth``."""
    if isinstance(value, dict) and "class" in value:
        name = value["class"]
        if depth >= max_depth:
            return f"{name}(...)"
        inner = ", ".join(
            f"{k}={_compact(v, depth + 1, max_depth)}"
            for k, v in value.items()
            if k != "class"
        )
        return f"{name}({inner})"
    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(
                f"{k}={_compact(v, depth, max_depth)}" for k, v in value.items()
            )
            + "}"
        )
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_compact(v, depth, max_depth) for v in value) + "]"
    if isinstance(value, str) and len(value) == 64:
        return value[:12] + "..."
    return str(value)


# ---------------------------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------------------------


def compare_stores(
    primary_a, primary_b, factories: Mapping[str, Any], classes=None
) -> Dict[str, Any]:
    """
    Compare the values of two closed stores, matched by identity. Returns the report as a plain
    (JSON-safe) dict; ``report["identical"]`` is whether they are value-identical in every class
    compared.
    """
    all_names = inventory_classes(factories)
    if classes is not None:
        unknown = [c for c in classes if c not in all_names]
        if len(unknown) > 0:
            raise ValueError(
                f"store_value_diff: {', '.join(unknown)} "
                f"{'is not a class' if len(unknown) == 1 else 'are not classes'} of the "
                f"inventory; the classes are {', '.join(all_names)}"
            )
    names = [n for n in all_names if classes is None or n in classes]

    a = read_store(primary_a, factories, names)
    b = read_store(primary_b, factories, names)
    describer = _Describer([a, b])

    report_classes: Dict[str, Any] = {}
    for name in names:
        ca = a.classes[name]
        cb = b.classes[name]

        def describe_row(identity, _name=name, _ca=ca, _cb=cb):
            found = _ca.records.get(identity) or _cb.records.get(identity)
            if found is None:
                return {"identity": identity}
            digest, key = found
            return {"identity": digest, "key": describer.key(_name, key)}

        def describe_value(match_key, _describe_row=describe_row):
            parent, named = match_key
            out = {"parent": _describe_row(parent)}
            for column, target, digest in named:
                out[column] = describer.reference(target, digest)
            # the value row's own identity: its parent's, and those of the rows it names
            out["identity"] = canonical_json(
                {
                    "parent": out["parent"].get("identity"),
                    "named": [[column, digest] for column, _target, digest in named],
                }
            )
            return out

        rows = _compare_tables(ca.rows, cb.rows, describe_row)
        values = {
            vt: _compare_tables(ca.values[vt], cb.values[vt], describe_value)
            for vt in sorted(ca.values)
        }
        unresolvable = sorted(
            {
                f"{table}.{column}"
                for read in (ca, cb)
                for table, cols in read.unresolvable_columns.items()
                for column in cols
            }
        )
        identical = (
            all(
                t["differ"] == 0
                and len(t["only_in_a"]) == 0
                and len(t["only_in_b"]) == 0
                and len(t["unmatched"]) == 0
                and len(t["columns_only_in_a"]) == 0
                and len(t["columns_only_in_b"]) == 0
                for t in [rows, *values.values()]
            )
            and len(unresolvable) == 0
        )
        report_classes[name] = {
            "identical": identical,
            "replicated": ca.replicated,
            "rows": rows,
            "values": values,
            "unresolvable_columns": unresolvable,
        }

    not_compared = sorted(set(a.not_compared) | set(b.not_compared))
    return {
        "a": str(a.primary),
        "b": str(b.primary),
        "classes": report_classes,
        "not_compared": not_compared,
        "identical": all(c["identical"] for c in report_classes.values())
        and len(not_compared) == 0,
    }


# ---------------------------------------------------------------------------------------------
# the text report
# ---------------------------------------------------------------------------------------------


def _fmt(x) -> str:
    if isinstance(x, str):
        return x
    return f"{x:.6g}"


def _format_table(lines, label, table, max_rows, indent) -> None:
    pad = " " * indent
    lines.append(
        f"{pad}{label}: {table['identical']} identical, {table['differ']} differ, "
        f"{len(table['only_in_a'])} only in A, {len(table['only_in_b'])} only in B, "
        f"{len(table['unmatched'])} cannot be matched by identity"
    )
    for side, key in (("A", "columns_only_in_a"), ("B", "columns_only_in_b")):
        for column in table[key]:
            lines.append(
                f"{pad}  column {column}: present in {side} only, not compared (the other "
                f"store's files do not hold it)"
            )
    for column, summary in table["columns"].items():
        text = f"{pad}  column {column}: {summary['count']} differ"
        if "max_abs_delta" in summary:
            text += (
                f"; max |delta| {_fmt(summary['max_abs_delta'])}, "
                f"max |delta|/|a| {_fmt(summary['max_rel_delta'])}"
            )
        if "json_keys" in summary:
            keys = list(summary["json_keys"].items())
            shown = ", ".join(f"{k} ({n})" for k, n in keys[:20])
            more = f", and {len(keys) - 20} more" if len(keys) > 20 else ""
            text += f"; JSON keys that differ: {shown}{more}"
        lines.append(text)
    for message in table["unmatched"][:max_rows]:
        lines.append(f"{pad}  cannot be matched: {message}")
    for side, entries in (("A", table["only_in_a"]), ("B", table["only_in_b"])):
        for entry in entries[:max_rows]:
            lines.append(f"{pad}  only in {side}: {_describe_text(entry)}")
    for entry in table["rows"][:max_rows]:
        lines.append(f"{pad}  differs: {_describe_text(entry)}")
        for column, diff in entry["columns"].items():
            if diff["kind"] == "float":
                lines.append(
                    f"{pad}    {column}: A {diff['a']} ({_fmt(diff['a_value'])}), "
                    f"B {diff['b']} ({_fmt(diff['b_value'])}), delta {_fmt(diff['delta'])}, "
                    f"|delta|/|a| {_fmt(diff['rel'])}"
                )
            elif diff["kind"] == "json":
                lines.append(
                    f"{pad}    {column} (JSON): keys {', '.join(diff['keys'])}"
                )
            else:
                lines.append(f"{pad}    {column}: A {diff['a']!r}, B {diff['b']!r}")
    hidden = len(table["rows"]) - max_rows
    if hidden > 0:
        lines.append(
            f"{pad}  ... and {hidden} more differing rows (--json has them all)"
        )


def _describe_text(entry: Mapping[str, Any]) -> str:
    if "parent" in entry:
        named = ", ".join(
            f"{k}={_compact(v)}"
            for k, v in entry.items()
            if k not in ("parent", "identity", "columns")
        )
        return f"value row of {_describe_text(entry['parent'])}" + (
            f" at {named}" if named else ""
        )
    identity = str(entry.get("identity", ""))
    key = entry.get("key")
    if key is None:
        return f"identity {identity[:16]}"
    return f"identity {identity[:16]} {_compact({'class': '', **key})}"


def format_report(report: Mapping[str, Any], max_rows: int = DEFAULT_MAX_ROWS) -> str:
    lines = [
        f"store_value_diff: A = {report['a']}",
        f"                  B = {report['b']}",
    ]
    classes = report["classes"]
    differ = [n for n, c in classes.items() if not c["identical"]]
    lines.append(
        f"{len(classes)} classes compared by identity: {len(classes) - len(differ)} "
        f"value-identical, {len(differ)} not"
    )
    for name, c in classes.items():
        state = "value-identical" if c["identical"] else "NOT value-identical"
        lines.append("")
        lines.append(f"{name}{' (replicated)' if c['replicated'] else ''}: {state}")
        _format_table(lines, "rows", c["rows"], max_rows, 2)
        for vt, table in c["values"].items():
            _format_table(lines, f"value table {vt}", table, max_rows, 2)
        for column in c["unresolvable_columns"]:
            lines.append(
                f"  column {column} names another row but declares no target: it cannot be "
                f"compared by identity, and is not compared by serial"
            )
    if len(report["not_compared"]) > 0:
        lines.append("")
        lines.append(
            f"tables not compared (no class of the inventory accounts for them): "
            f"{', '.join(report['not_compared'])}"
        )
    lines.append("")
    lines.append(
        "RESULT: the stores are value-identical"
        if report["identical"]
        else "RESULT: the stores are NOT value-identical"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None, factories=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.store_value_diff",
        description="Compare the values of two closed stores, matched by physical identity.",
    )
    parser.add_argument("a", help="the first store's primary (.sqlite)")
    parser.add_argument("b", help="the second store's primary (.sqlite)")
    parser.add_argument(
        "--json", action="store_true", help="write the report as JSON instead of text"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        metavar="NAME",
        help="compare only these classes of the inventory (all of them by default)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help=f"differing rows printed per table in the text report (default {DEFAULT_MAX_ROWS})",
    )
    args = parser.parse_args(argv)

    if factories is None:
        from config.datastore import factories

    try:
        report = compare_stores(args.a, args.b, factories, classes=args.classes)
    except (RuntimeError, ValueError) as e:
        print(f"store_value_diff: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=1, sort_keys=False, allow_nan=False))
    else:
        print(format_report(report, max_rows=args.max_rows))
    return 0 if report["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
