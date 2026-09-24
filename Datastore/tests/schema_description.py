"""
A deterministic description of the schema ``Datastore._build_schema`` builds, for the witness
``Datastore/tests/data/schema_at_base.json``.

The witness was captured from the code **before** the schema builder moved into
``Datastore/SQL/schema.py`` (store-fingerprint prompt 01; its log records the command and the
SHA). The tests compare this description of what the code builds now against that file. The
file is never regenerated to make a test pass: it is the evidence that the move changed nothing.

This module imports only the standard library and ``sqlalchemy``. It does not import ``Datastore/SQL/schema.py``
or ``Datastore/SQL/Datastore.py`` at module scope, so that it can be run against a checkout of the
code before the change existed, to re-capture the witness independently::

    git archive <base sha> | tar -x -C <dir>
    cd <dir> && PYTHONPATH=. <repo>/venv/bin/python <repo>/Datastore/tests/schema_description.py out.json

This module is not a test module (no ``test_`` prefix); the test modules import it.
"""

import importlib
import json
import sys
from typing import Any, Dict, Mapping

import sqlalchemy as sqla
from sqlalchemy.dialects import sqlite
from sqlalchemy.schema import CreateIndex, CreateTable

# bumped only if the shape of the description changes, which would need a new witness captured
# from the base code, never one regenerated from the new code
DESCRIPTION_FORMAT = 1

_DIALECT = sqlite.dialect()


def _name(value) -> Any:
    """A constraint's or index's name, or None for an unnamed one (SQLAlchemy's sentinel)."""
    return str(value) if isinstance(value, str) else None


def _default(column: sqla.Column) -> Any:
    if column.default is None:
        return None
    arg = getattr(column.default, "arg", None)
    if callable(arg):
        return "<callable>"
    return repr(arg)


def describe_column(column: sqla.Column) -> Dict[str, Any]:
    return {
        "name": column.name,
        "type": repr(column.type),
        "sqlite_type": column.type.compile(dialect=_DIALECT),
        "nullable": column.nullable,
        "primary_key": column.primary_key,
        "foreign_keys": sorted(
            (
                {
                    "target": fk.target_fullname,
                    "ondelete": fk.ondelete,
                    "onupdate": fk.onupdate,
                }
                for fk in column.foreign_keys
            ),
            key=lambda d: json.dumps(d, sort_keys=True),
        ),
        "index": column.index,
        "unique": column.unique,
        "default": _default(column),
        "server_default": (
            None if column.server_default is None else repr(column.server_default)
        ),
        "autoincrement": repr(column.autoincrement),
    }


def describe_constraint(constraint) -> Dict[str, Any]:
    out = {
        "kind": type(constraint).__name__,
        "name": _name(constraint.name),
        "columns": [c.name for c in constraint.columns],
    }
    if isinstance(constraint, sqla.ForeignKeyConstraint):
        out["targets"] = [e.target_fullname for e in constraint.elements]
    if isinstance(constraint, sqla.CheckConstraint):
        out["sqltext"] = str(constraint.sqltext)
    return out


def describe_table(table: sqla.Table) -> Dict[str, Any]:
    indexes = sorted(table.indexes, key=lambda i: str(i.name))
    return {
        "name": table.name,
        "columns": [describe_column(c) for c in table.columns],
        "constraints": sorted(
            (describe_constraint(c) for c in table.constraints),
            key=lambda d: json.dumps(d, sort_keys=True),
        ),
        "indexes": [
            {
                "name": _name(i.name),
                "columns": [c.name for c in i.columns],
                "unique": i.unique,
            }
            for i in indexes
        ],
        "ddl": str(CreateTable(table).compile(dialect=_DIALECT)).strip(),
        "index_ddl": [
            str(CreateIndex(i).compile(dialect=_DIALECT)).strip() for i in indexes
        ],
    }


def _describe_value(value) -> Any:
    if isinstance(value, sqla.Table):
        return {"table": value.name}
    if isinstance(value, sqla.Column):
        return {
            "column": value.name,
            "of_table": None if value.table is None else value.table.name,
        }
    if isinstance(value, (list, tuple)):
        return [_describe_value(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"undescribed schema record value of type {type(value).__name__}")


def describe_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """A schema record's non-callable fields. Callables (the actor's inserter) are left out; a
    field that is ``None`` (the inserter of a class with no table) is kept."""
    return {
        key: _describe_value(value)
        for key, value in sorted(record.items())
        if value is None or not callable(value)
    }


def describe_schema(
    records: Mapping[str, Mapping[str, Any]],
    tables: Mapping[str, sqla.Table],
    metadata: sqla.MetaData,
) -> Dict[str, Any]:
    """Everything the schema builder builds, apart from the inserters."""
    classes = {}
    for cls_name, record in records.items():
        table = record.get("table")
        classes[cls_name] = {
            "has_table": table is not None,
            "record": describe_record(record),
            "table": None if table is None else describe_table(table),
        }
    return {
        "format": DESCRIPTION_FORMAT,
        "class_order": list(records.keys()),
        "classes": classes,
        "table_order": list(tables.keys()),
        "metadata_tables": sorted(metadata.tables.keys()),
    }


def dumps(description: Mapping[str, Any]) -> str:
    return json.dumps(description, sort_keys=True, indent=1) + "\n"


def actor_with_built_schema():
    """
    A ``Datastore`` actor instance on which ``_build_schema`` has run, and nothing else.

    The actor class is reached without Ray as ``Datastore.__ray_metadata__.modified_class``. Its
    ``__init__`` is **not** called, because it opens and writes a database file. The instance is
    given only the attributes ``__init__`` sets before ``_build_schema`` and that
    ``_build_schema`` reads: the factories (registered exactly as ``__init__`` registers them), a
    fresh ``MetaData`` (as ``_create_engine`` makes it), and empty ``_tables``, ``_inserters`` and
    ``_schema``.
    """
    # the module, by its full name: Datastore/SQL/__init__.py rebinds ``Datastore.SQL.Datastore``
    # to the actor class, so ``from Datastore.SQL import Datastore`` would not give the module
    datastore_module = importlib.import_module("Datastore.SQL.Datastore")

    cls = datastore_module.Datastore.__ray_metadata__.modified_class
    actor = object.__new__(cls)
    actor._factories = {}
    actor.register_factories(datastore_module._factories)
    actor._metadata = sqla.MetaData()
    actor._tables = {}
    actor._inserters = {}
    actor._schema = {}
    actor._build_schema()
    return actor


def describe_actor_schema() -> Dict[str, Any]:
    actor = actor_with_built_schema()
    return describe_schema(actor._schema, actor._tables, actor._metadata)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <output.json>")
    text = dumps(describe_actor_schema())
    with open(sys.argv[1], "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
