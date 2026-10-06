"""
The one schema builder: the ``sqla.Table`` objects and per-class schema records for a set of
storable-class factories.

``Datastore._build_schema`` calls ``build_schema`` and adds only the inserters, which are bound to
the actor's ``_insert`` and so stay in the actor. The read-only store reader
(``Datastore/store_reader.py``) calls it too, with no actor. There is one definition of what the
tables are; do not copy this loop.

Building the schema writes nothing: the tables are declared into ``metadata`` and never created
here. Creating missing tables is ``Datastore._ensure_tables``' job, on a read-write engine.

**The one comparison** of a file against the declared tables is here too
(prompts/datastore-generic, prompt 05). ``schema_differences`` compares one SQLite file's table
and column names with a mapping of declared tables, reading only ``sqlite_master`` and ``PRAGMA
table_info``, and ``StoreSchemaMismatch`` is the one refusal of a file that differs, naming the
file and each table and column by kind. The store reader, the read-write check at open and both pool
constructors' check of the primary use them; do not write a second comparison.

**The order of a drop** is here too (prompts/datastore-generic, prompt 06). ``drop_order`` puts a
set of tables in an order in which each can be dropped while the others are present, derived from
their foreign keys alone; the actor drops the tables it is given in that order. No order is kept by
hand.

**The replication facts a factory declares** are carried here too (prompts/datastore-generic,
prompt 07). Three keys of ``register()``, each checked against the class's own table and copied
into the record of every class with a table: ``owner_column`` (the column whose one foreign key
names the row this table's rows belong to), ``monotone_flags`` (the Boolean columns a get may turn
on and nothing turns off) and ``validated_column`` (the Boolean column of the class's validated
flag, which the factory's ``revalidate`` recomputes from stored rows). A declaration that does not
fit its table raises ``ValueError``.

This module itself imports only ``sqlalchemy`` and ``Datastore.contract``, which names the version
table every prepended ``version`` column references. Importing it runs
``Datastore/SQL/__init__.py``, which imports the actor module and so ``ray``; that initialises
nothing.
"""

from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Tuple

import sqlalchemy as sqla

from Datastore.contract import VERSION_TABLE


class BuiltSchema(NamedTuple):
    """What ``build_schema`` returns: class name -> table, for each class that has one, and class
    name -> schema record, for every class. Both are in the factories' order."""

    tables: Dict[str, sqla.Table]
    records: Dict[str, Dict[str, Any]]


def build_schema(metadata: sqla.MetaData, factories: Mapping[str, Any]) -> BuiltSchema:
    """
    Build a table in ``metadata`` for each factory whose ``register()`` asks for one, and a schema
    record for every factory.

    A record holds the class name and its ``validate_on_startup`` flag. For a class with a table
    it also holds which of the prepended columns it uses (``use_serial``, ``use_version``,
    ``use_timestamp``, ``use_stepping`` and ``stepping_mode``), those columns, the factory's own
    columns and the table. The prepended columns come first, in the order ``serial``, ``version``,
    ``timestamp``, ``stepping``, then the factory's columns in the order it gives them. A class
    whose ``register()`` returns ``None`` has ``"table": None`` and no table.

    A record of a class with a table also holds the three replication facts its factory declares,
    each checked against the table (the module docstring): ``owner_column`` (``None`` when not
    declared), ``monotone_flags`` (a tuple, ``()`` when not declared) and ``validated_column``
    (``None`` when not declared). A declaration that does not fit raises ``ValueError`` naming the
    class, the key and what is wrong. A record of a class with no table does not hold them, so a
    reader uses ``.get``.

    The records hold no inserter: the actor adds its own.
    """
    tables: Dict[str, sqla.Table] = {}
    records: Dict[str, Dict[str, Any]] = {}

    # iterate through all registered storage adapters, querying them for the columns
    # they need to persist their data
    for cls_name, factory in factories.items():
        if cls_name in records:
            raise RuntimeWarning(
                f"Duplicate registered factory for storable class '{cls_name}'"
            )

        # query class for a list of columns that it wants to store
        registration_data = factory.register()

        # does this storage object require its own table?
        if registration_data is None:
            records[cls_name] = {
                "name": cls_name,
                "validate_on_startup": False,
                "table": None,
            }
            continue

        schema = {
            "name": cls_name,
            "validate_on_startup": registration_data.get("validate_on_startup", False),
        }

        # generate main table for this adapter class
        tab = sqla.Table(
            cls_name,
            metadata,
        )

        use_serial = registration_data.get("serial", True)
        schema["use_serial"] = use_serial
        if use_serial:
            serial_col = sqla.Column("serial", sqla.Integer, primary_key=True)
            tab.append_column(serial_col)
            schema["serial_col"] = serial_col

        # attach pre-defined columns
        use_version = registration_data.get("version", False)
        schema["use_version"] = use_version
        if use_version:
            version_col = sqla.Column(
                "version",
                sqla.Integer,
                sqla.ForeignKey(f"{VERSION_TABLE}.serial"),
                index=True,
            )
            tab.append_column(version_col)
            schema["version_col"] = version_col

        use_timestamp = registration_data.get("timestamp", False)
        schema["use_timestamp"] = use_timestamp
        if use_timestamp:
            timestamp_col = sqla.Column("timestamp", sqla.DateTime())
            tab.append_column(timestamp_col)
            schema["timestamp_col"] = timestamp_col

        use_stepping = registration_data.get("stepping", False)
        if isinstance(use_stepping, str):
            if use_stepping not in ["minimum", "exact"]:
                print(
                    f"!! Warning: ignored stepping selection '{use_stepping}' when registering storable class factory for '{cls_name}'"
                )
                use_stepping = False

        _use_stepping = isinstance(use_stepping, str) or use_stepping is True
        schema["use_stepping"] = _use_stepping
        if _use_stepping:
            stepping_col = sqla.Column("stepping", sqla.Integer)
            tab.append_column(stepping_col)
            schema["stepping_col"] = stepping_col

            _stepping_mode = None if not isinstance(use_stepping, str) else use_stepping
            schema["stepping_mode"] = _stepping_mode

        # append all columns supplied by the class
        sqla_columns = registration_data.get("columns", [])
        for col in sqla_columns:
            tab.append_column(col)
        schema["columns"] = sqla_columns

        # the replication facts the factory declares (prompts/datastore-generic, prompt 07), each
        # checked against the table's own columns
        schema["owner_column"] = _declared_owner_column(
            cls_name, tab, registration_data
        )
        schema["monotone_flags"] = _declared_monotone_flags(
            cls_name, tab, registration_data
        )
        schema["validated_column"] = _declared_validated_column(
            cls_name, tab, registration_data
        )

        # store in table cache
        schema["table"] = tab

        tables[cls_name] = tab
        records[cls_name] = schema

    return BuiltSchema(tables=tables, records=records)


def _refuse_declaration(cls_name: str, key: str, value, tab: sqla.Table, why: str):
    """The ``ValueError`` of a declaration that does not fit its table: the class, the key, the
    value declared and what is wrong, with the table's columns."""
    return ValueError(
        f'build_schema: the factory of "{cls_name}" declares {key}={value!r}, {why}; the columns '
        f"of its table are {[c.name for c in tab.columns]}"
    )


def _declared_owner_column(
    cls_name: str, tab: sqla.Table, registration_data: Mapping[str, Any]
) -> Optional[str]:
    """``owner_column``: a column of the table with exactly one foreign key, which names the row
    this table's rows belong to; ``None`` when it is not declared."""
    owner = registration_data.get("owner_column", None)
    if owner is None:
        return None
    if not isinstance(owner, str) or owner not in tab.c:
        raise _refuse_declaration(
            cls_name, "owner_column", owner, tab, "which is not a column of its table"
        )
    n = len(tab.c[owner].foreign_keys)
    if n != 1:
        raise _refuse_declaration(
            cls_name,
            "owner_column",
            owner,
            tab,
            f"a column with {n} foreign keys, where the column naming a row's owner has exactly "
            "one",
        )
    return owner


def _declared_monotone_flags(
    cls_name: str, tab: sqla.Table, registration_data: Mapping[str, Any]
) -> Tuple[str, ...]:
    """``monotone_flags``: Boolean columns of the table that a get may turn on and nothing turns
    off, as a tuple; ``()`` when it is not declared."""
    declared = registration_data.get("monotone_flags", ())
    if isinstance(declared, str):
        raise _refuse_declaration(
            cls_name,
            "monotone_flags",
            declared,
            tab,
            "a string, where a sequence of column names is expected",
        )
    flags = tuple(declared)
    absent = [f for f in flags if not isinstance(f, str) or f not in tab.c]
    if len(absent) > 0:
        raise _refuse_declaration(
            cls_name,
            "monotone_flags",
            flags,
            tab,
            f"and {absent} are not columns of its table",
        )
    not_boolean = [f for f in flags if not isinstance(tab.c[f].type, sqla.Boolean)]
    if len(not_boolean) > 0:
        raise _refuse_declaration(
            cls_name,
            "monotone_flags",
            flags,
            tab,
            f"and {not_boolean} are not Boolean columns",
        )
    return flags


def _declared_validated_column(
    cls_name: str, tab: sqla.Table, registration_data: Mapping[str, Any]
) -> Optional[str]:
    """``validated_column``: the Boolean column of the class's validated flag, which its
    factory's ``revalidate`` recomputes from stored rows; ``None`` when it is not declared.
    """
    column = registration_data.get("validated_column", None)
    if column is None:
        return None
    if not isinstance(column, str) or column not in tab.c:
        raise _refuse_declaration(
            cls_name,
            "validated_column",
            column,
            tab,
            "which is not a column of its table",
        )
    if not isinstance(tab.c[column].type, sqla.Boolean):
        raise _refuse_declaration(
            cls_name,
            "validated_column",
            column,
            tab,
            f"a column of type {tab.c[column].type!r}, which is not Boolean",
        )
    return column


def drop_order(tables: Iterable[sqla.Table]) -> List[sqla.Table]:
    """
    The given tables in an order in which each can be dropped while the others not yet dropped are
    present: a table that declares a foreign key into another of the given tables comes before it.

    Derived from the tables' foreign keys alone. A foreign key into a table that is not given, or
    into the table itself, does not constrain the order. Among the tables free to be dropped next,
    the one given first comes first, so the order is fixed by the input. A cycle among the given
    tables has no such order, and raises ``ValueError`` naming the tables on or behind it.
    """
    given: List[sqla.Table] = []
    for table in tables:
        if table not in given:
            given.append(table)

    # for each table, the other given tables that declare a foreign key into it: it can be
    # dropped only once every one of those has been
    referenced_by: Dict[sqla.Table, set] = {table: set() for table in given}
    for table in given:
        for fk in table.foreign_keys:
            target = fk.column.table
            if target is not table and target in referenced_by:
                referenced_by[target].add(table)

    ordered: List[sqla.Table] = []
    remaining = list(given)
    while len(remaining) > 0:
        free = [t for t in remaining if referenced_by[t].isdisjoint(remaining)]
        if len(free) == 0:
            raise ValueError(
                "drop_order: the foreign keys among the tables "
                f"{[t.name for t in remaining]} form a cycle, so no order drops each while "
                "the others are present"
            )
        ordered.append(free[0])
        remaining.remove(free[0])

    return ordered


class SchemaDifferences(NamedTuple):
    """
    What ``schema_differences`` found: how one SQLite file differs from the tables it was compared
    with, by kind.

    - ``absent_tables``: the declared tables the file lacks, in declaration order;
    - ``extra_tables``: the tables the file has that are not declared, in file order;
    - ``absent_columns``: for each declared table the file has, the declared columns it lacks, in
      declaration order. A table that lacks none is not a key;
    - ``extra_columns``: for each declared table the file has, the file's columns that are not
      declared, in file order. A table with none is not a key.
    """

    absent_tables: Tuple[str, ...]
    extra_tables: Tuple[str, ...]
    absent_columns: Dict[str, Tuple[str, ...]]
    extra_columns: Dict[str, Tuple[str, ...]]

    @property
    def empty(self) -> bool:
        """True iff the file holds exactly the declared tables, each with exactly its declared
        columns."""
        return (
            len(self.absent_tables) == 0
            and len(self.extra_tables) == 0
            and len(self.absent_columns) == 0
            and len(self.extra_columns) == 0
        )

    def describe(self) -> str:
        """Every difference, by kind, naming each table and column, in one clause list."""

        def names(items) -> str:
            return ", ".join(f'"{n}"' for n in items)

        parts = []
        if len(self.absent_tables) > 0:
            parts.append(f"it lacks the table(s) {names(self.absent_tables)}")
        if len(self.extra_tables) > 0:
            parts.append(
                f"it has the table(s) {names(self.extra_tables)}, which this code does not "
                "declare"
            )
        for table, columns in self.absent_columns.items():
            parts.append(f'its table "{table}" lacks the column(s) {names(columns)}')
        for table, columns in self.extra_columns.items():
            parts.append(
                f'its table "{table}" has the column(s) {names(columns)}, which this code does '
                "not declare"
            )
        return "; ".join(parts)


def schema_differences(conn, tables: Mapping[str, sqla.Table]) -> SchemaDifferences:
    """
    Compare one SQLite file, through ``conn`` (a SQLAlchemy connection to it), with ``tables``
    (name -> ``sqla.Table``): its table names, from ``sqlite_master`` (type ``table``, without
    SQLite's own ``sqlite_`` tables), and the column names of each declared table it has, from
    ``PRAGMA table_info``. Not types, nullability, keys, defaults or indexes.

    It reads only ``sqlite_master`` and ``PRAGMA table_info``, and writes nothing.
    """
    in_file = [
        row[0]
        for row in conn.exec_driver_sql(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
        if not row[0].startswith("sqlite_")
    ]
    present = set(in_file)

    absent_columns: Dict[str, Tuple[str, ...]] = {}
    extra_columns: Dict[str, Tuple[str, ...]] = {}
    for name, table in tables.items():
        if name not in present:
            continue
        quoted = name.replace('"', '""')
        file_columns = [
            row[1] for row in conn.exec_driver_sql(f'PRAGMA table_info("{quoted}")')
        ]
        declared = [c.name for c in table.columns]
        absent = tuple(c for c in declared if c not in file_columns)
        extra = tuple(c for c in file_columns if c not in declared)
        if len(absent) > 0:
            absent_columns[name] = absent
        if len(extra) > 0:
            extra_columns[name] = extra

    return SchemaDifferences(
        absent_tables=tuple(n for n in tables if n not in present),
        extra_tables=tuple(n for n in in_file if n not in tables),
        absent_columns=absent_columns,
        extra_columns=extra_columns,
    )


class StoreSchemaMismatch(RuntimeError):
    """
    A file of a sharded store does not hold exactly the tables and columns the code declares, so
    the store is refused before anything reads its tables (prompts/datastore-generic, prompt 05).

    ``verb`` is the operation that refused ("read" from the read-only reader, "open" from a pool's
    constructor), ``primary`` the store's primary, ``file`` a description of the file that differs
    (``shard #N "<path>"`` or ``the primary "<path>"``), and ``differences`` the
    ``SchemaDifferences`` found. It is raised where it is built, and never wrapped.
    """

    def __init__(self, verb: str, primary, file: str, differences: SchemaDifferences):
        self.verb = verb
        self.primary = primary
        self.file = file
        self.differences = differences
        super().__init__(self._message())

    def _message(self) -> str:
        return (
            f'Cannot {self.verb} sharded datastore "{str(self.primary)}": {self.file} differs '
            f"from the tables this code declares ({self.differences.describe()}), so nothing "
            "was read"
        )

    def __str__(self) -> str:
        return self._message()

    def __reduce__(self):
        return (
            type(self),
            (self.verb, self.primary, self.file, self.differences),
        )
