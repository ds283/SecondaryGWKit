"""
The one schema builder: the ``sqla.Table`` objects and per-class schema records for a set of
storable-class factories.

``Datastore._build_schema`` calls ``build_schema`` and adds only the inserters, which are bound to
the actor's ``_insert`` and so stay in the actor. The read-only store reader
(``Datastore/store_reader.py``) calls it too, with no actor. There is one definition of what the
tables are; do not copy this loop.

Building the schema writes nothing: the tables are declared into ``metadata`` and never created
here. Creating missing tables is ``Datastore._ensure_tables``' job, on a read-write engine.

This module itself imports only ``sqlalchemy``. Importing it runs ``Datastore/SQL/__init__.py``,
which imports the actor module and so ``ray``; that initialises nothing.
"""

from typing import Any, Dict, Mapping, NamedTuple

import sqlalchemy as sqla


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
                sqla.ForeignKey("version.serial"),
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

        # store in table cache
        schema["table"] = tab

        tables[cls_name] = tab
        records[cls_name] = schema

    return BuiltSchema(tables=tables, records=records)
