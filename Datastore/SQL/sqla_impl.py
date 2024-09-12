import functools
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Optional, Union, Mapping, Callable

import sqlalchemy as sqla
from ray import remote

from Datastore.SQL.ObjectFactories.LambdaCDM import sqla_LambdaCDM_factory
from Datastore.SQL.ObjectFactories.MatterTransferFunction import (
    sqla_MatterTransferFunctionIntegration_factory,
    sqla_MatterTransferFunctionValue_factory,
    sqla_MatterTransferFunctionContainer_factory,
)
from Datastore.SQL.ObjectFactories.TensorGreenFunction import (
    sqla_TensorGreenFunctionIntegration_factory,
    sqla_TensorGreenFunctionValue_factory,
    sqla_TensorGreenFunctionContainer_factory,
)
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from Datastore.SQL.ObjectFactories.integration_metadata import (
    sqla_IntegrationSolver_factory,
)
from Datastore.SQL.ObjectFactories.redshift import sqla_redshift_factory
from Datastore.SQL.ObjectFactories.tolerance import sqla_tolerance_factory
from Datastore.SQL.ObjectFactories.wavenumber import (
    sqla_wavenumber_factory,
    sqla_wavenumber_exit_time_factory,
)
from utilities import WallclockTimer

VERSION_ID_LENGTH = 64


PathType = Union[str, PathLike]


_adapters = {
    "redshift": sqla_redshift_factory,
    "wavenumber": sqla_wavenumber_factory,
    "wavenumber_exit_time": sqla_wavenumber_exit_time_factory,
    "tolerance": sqla_tolerance_factory,
    "LambdaCDM": sqla_LambdaCDM_factory,
    "IntegrationSolver": sqla_IntegrationSolver_factory,
    "MatterTransferFunctionIntegration": sqla_MatterTransferFunctionIntegration_factory,
    "MatterTransferFunctionValue": sqla_MatterTransferFunctionValue_factory,
    "MatterTransferFunctionContainer": sqla_MatterTransferFunctionContainer_factory,
    "TensorGreenFunctionIntegration": sqla_TensorGreenFunctionIntegration_factory,
    "TensorGreenFunctionValue": sqla_TensorGreenFunctionValue_factory,
    "TensorGreenFunctionContainer": sqla_TensorGreenFunctionContainer_factory,
}

_FactoryMappingType = Mapping[str, SQLAFactoryBase]
_TableMappingType = Mapping[str, sqla.Table]
_InserterMappingType = Mapping[str, Callable]


@remote
class Datastore:
    def __init__(self, version_id: str):
        """
        Initialize an SQL datastore object
        :param version_id: version identifier used to tag results written in to the store
        """

        # initialize SQLAlchemy objects
        self._engine: Optional[sqla.Engine] = None
        self._metadata: sqla.Metadata = None

        # store version code and initialize (as blank) the corresponding serial code
        self._version_id = version_id
        self._version_serial = None

        # initialize set of registered storable class adapters
        self._factories: _FactoryMappingType = {}
        self.register_factories(_adapters)

        # initialize empty dict of storage schema
        # each record collects SQLAlchemy column and table definitions, queries, etc., for a registered storable class factories
        self._tables: _TableMappingType = {}
        self._inserters: _InserterMappingType = {}
        self._schema = {}

    def _create_engine(
        self, db_name: PathType, expect_exists: bool = False, timeout=None
    ):
        """
        Create and initialize an SQLAlchemy engine corresponding to the name data container, but does not physically initialize the
        container. Use create_datastore() for this purpose if needed.
        :param db_name: path to on-disk database
        :param expect_exists: should we expect the database to already exist? Ensures its content is not overwritten.
        :return:
        """
        db_file = Path(db_name).resolve()

        if not expect_exists and db_file.exists():
            raise RuntimeError(
                "Specified database cache {path} already exists".format(path=db_name)
            )

        if expect_exists and not db_file.exists():
            raise RuntimeError(
                "Specified database cache {path} does not exist; please create using "
                "--create-database".format(path=db_name)
            )

        if db_file.is_dir():
            raise RuntimeError(
                "Specified database cache {path} is a directory".format(path=db_name)
            )

        # if file does not exist, ensure its parent directories exist
        if not db_file.exists():
            db_file.parents[0].mkdir(exist_ok=True, parents=True)

        connect_args = {}
        if timeout is not None:
            connect_args["timeout"] = timeout

        self._engine = sqla.create_engine(
            f"sqlite:///{db_name}",
            future=True,
            connect_args=connect_args,
        )
        self._metadata = sqla.MetaData()

        self._version_table = sqla.Table(
            "versions",
            self._metadata,
            sqla.Column("serial", sqla.Integer, primary_key=True, nullable=False),
            sqla.Column("version_id", sqla.String(VERSION_ID_LENGTH)),
        )

        self._build_storage_schema()

    def create_datastore(self, db_name: PathType, timeout=None):
        """
        Create and initialize an empty data container. Assumes the container not to be physically present at the specified path
        and will fail with an error if it is
        :param db_name: path to on-disk database
        :return:
        """
        self._ensure_no_engine()
        self._create_engine(db_name, expect_exists=False, timeout=timeout)

        print("-- creating database tables")

        # generate internal tables
        self._version_table.create(self._engine)
        self._version_serial = self._make_version_serial()

        # generate tables defined by any registered storable classes
        self._create_storage_tables()

    def _make_version_serial(self):
        """
        Normalize the supplied version label into a database rowid
        :return:
        """
        self._ensure_engine()

        with self._engine.begin() as conn:
            # query database for serial number corresponding to this version label
            serial = conn.execute(
                sqla.select(self._version_table.c.serial).filter(
                    self._version_table.c.version_id == self._version_id
                )
            ).scalar()

            if serial is not None:
                return serial

            # no existing serial number was found, so insert a new one
            serial = conn.execute(
                sqla.select(sqla.func.max(self._version_table.c.serial)).select_from(
                    self._version_table
                )
            ).scalar()

            # if .scalar returns an empty result, there are no serial numbers currently in use
            if serial is None:
                serial = -1

            # insert a new database row for this version label
            conn.execute(
                sqla.insert(self._version_table),
                {"serial": serial + 1, "version_id": self._version_id},
            )

            conn.commit()

        return serial

    def _ensure_engine(self):
        if self._engine is None:
            raise RuntimeError(f"No storage engine has been initialized")

    def _ensure_no_engine(self):
        if self._engine is not None:
            raise RuntimeError(f"A storage engine has already been initialized")

    def open_datastore(self, db_name: str, timeout=None) -> None:
        self._ensure_no_engine()
        self._create_engine(db_name, expect_exists=True, timeout=timeout)

        self._version_serial = self._make_version_serial()

    def register_factories(self, factories: _FactoryMappingType):
        """
        Register a factory for a storable class. Factories are delegates for the SQLAlchemy operations
        needed to serialize and deserialize storable classes into the Datastore.
        These are factories, not adapters. They don't wrap the storable classes themselves.
        This is a deliberate design decision; everything to do with database I/O is supposed to happen
        on the node running the Datastore actor, for performance reasons.
        :param factories:
        :return:
        """
        if not isinstance(factories, Mapping):
            raise RuntimeError("Expecting factory_set to be a mapping instance")

        for cls_name, factory in factories.items():
            if cls_name in self._factories:
                raise RuntimeWarning(
                    f"Duplicate attempt to register storable class factory '{cls_name}'"
                )

            self._factories[cls_name] = factory
            print(f"Registered storable class factory '{cls_name}'")

    def _build_storage_schema(self):
        # iterate through all registered storage adapters, querying them for the columns
        # they need to persist their data
        for cls_name, factory in self._factories.items():
            if cls_name in self._schema:
                raise RuntimeWarning(
                    f"Duplicate registered factory for storable class '{cls_name}'"
                )

            schema = {"name": cls_name}

            # query class for a list of columns that it wants to store
            table_data = factory.generate_columns()

            # does this storage object require its own table?
            if table_data is not None:
                # generate main table for this adapter class
                serial_col = sqla.Column("serial", sqla.Integer, primary_key=True)
                tab = sqla.Table(
                    cls_name,
                    self._metadata,
                    serial_col,
                )

                schema["serial_col"] = serial_col

                # attach pre-defined columns
                use_version = table_data.get("version", False)
                schema["use_version"] = use_version
                if use_version:
                    version_col = sqla.Column(
                        "version", sqla.Integer, sqla.ForeignKey("versions.serial")
                    )
                    tab.append_column(version_col)
                    schema["version_col"] = version_col

                use_timestamp = table_data.get("timestamp", False)
                schema["use_timestamp"] = use_timestamp
                if use_timestamp:
                    timestamp_col = sqla.Column("timestamp", sqla.DateTime())
                    tab.append_column(timestamp_col)
                    schema["timestamp_col"] = timestamp_col

                use_stepping = table_data.get("stepping", False)
                if isinstance(use_stepping, str):
                    if use_stepping not in ["minimum", "exact"]:
                        print(
                            f"Warning: ignored stepping selection '{use_stepping}' when registering storable class factory for '{cls_name}'"
                        )
                        use_stepping = False

                _use_stepping = isinstance(use_stepping, str) or use_stepping is True
                schema["use_stepping"] = _use_stepping
                if _use_stepping:
                    stepping_col = sqla.Column("stepping", sqla.Integer)
                    tab.append_column(stepping_col)
                    schema["stepping_col"] = stepping_col

                    _stepping_mode = (
                        None if not isinstance(use_stepping, str) else use_stepping
                    )
                    schema["stepping_mode"] = _stepping_mode

                # append all columns supplied by the class
                sqla_columns = table_data.get("columns", [])
                for col in sqla_columns:
                    tab.append_column(col)
                schema["columns"] = sqla_columns

                # store in table cache
                schema["table"] = tab

                # build inserter
                inserter = functools.partial(self._insert, schema, tab)
                schema["insert"] = inserter

                # also store table and inserter in their own separate cache
                self._tables[cls_name] = tab
                self._inserters[cls_name] = inserter

                print(
                    f"Registered storage schema for storable class adapter '{cls_name}' with database table '{tab.name}'"
                )
            else:
                schema["table"] = None
                schema["insert"] = None

                print(
                    f"Registered storage scheme for storable class adapter '{cls_name}' without database table"
                )

            self._schema[cls_name] = schema

    def _create_storage_tables(self):
        self._ensure_engine()

        for tab in self._tables.values():
            tab.create(self._engine)

    def _ensure_registered_schema(self, cls_name: str):
        if cls_name not in self._factories:
            raise RuntimeError(
                f'No storable class of type "{cls_name}" has been registered'
            )

    def object_get(self, ObjectClass, **kwargs):
        with WallclockTimer() as timer:
            self._ensure_engine()

            if isinstance(ObjectClass, str):
                cls_name = ObjectClass
            else:
                cls_name = ObjectClass.__name__
            self._ensure_registered_schema(cls_name)

            record = self._schema[cls_name]

            tab = record["table"]
            inserter = record["insert"]

            # obtain type of factory class for this storable
            factory = self._factories[cls_name]

            if "payload_data" in kwargs:
                payload_data = kwargs["payload_data"]
                scalar = False

                payload_size = len(payload_data)
                # print(
                #     f'** Datastore.object_get() starting for object group of class "{cls_name}"'
                # )
                # print(f"**   payload size = {payload_size} items")
            else:
                payload_data = [kwargs]
                scalar = True

                # print(
                #     f'** Datastore.object_get() starting for scalar query of class "{cls_name}"'
                # )

            with self._engine.begin() as conn:
                objects = [
                    factory.build(
                        payload=p,
                        conn=conn,
                        table=tab,
                        inserter=inserter,
                        tables=self._tables,
                        inserters=self._inserters,
                    )
                    for p in payload_data
                ]
                conn.commit()

        # if scalar:
        #     print(
        #         f'** Datastore.object_get() finished in time {timer.elapsed:.3g} sec for scalar query of class "{cls_name}"'
        #     )
        # else:
        #     print(
        #         f'** Datastore.object_get() finsihed in time {timer.elapsed:.3g} sec for size={payload_size} object group of class "{cls_name}" = {float(timer.elapsed)/payload_size:.3g} sec per item'
        #     )

        if scalar:
            return objects[0]

        return objects

    def object_store(self, objects):
        with WallclockTimer() as timer:
            self._ensure_engine()

            # print(f"** Datastore.object_store() starting")

            if isinstance(objects, list) or isinstance(objects, tuple):
                payload_data = objects
                scalar = False
                # print(f" **   payload size = {len(objects)} items")
            else:
                payload_data = [objects]
                scalar = True

            output_objects = []
            with self._engine.begin() as conn:
                for obj in payload_data:
                    cls_name = type(objects).__name__
                    self._ensure_registered_schema(cls_name)

                    with WallclockTimer() as store_timer:
                        # print(f'**   starting store for object of type "{cls_name}"')

                        record = self._schema[cls_name]

                        tab = record["table"]
                        inserter = record["insert"]

                        factory = self._factories[cls_name]

                        output_objects.append(
                            factory.store(
                                objects,
                                conn=conn,
                                table=tab,
                                inserter=inserter,
                                tables=self._tables,
                                inserters=self._inserters,
                            )
                        )

                    # print(
                    #     f'**   finished store for object of type "{cls_name}" in time {store_timer.elapsed:.3g} sec'
                    # )

                conn.commit()

        # print(
        #     f'** Datastore.object_store() finished in time {timer.elapsed:.3g} sec'
        # )

        if scalar:
            return output_objects[0]

        return output_objects

    def _insert(self, schema, table, conn, payload):
        if table is None:
            raise RuntimeError(f"Attempt to insert into null table (scheme='{schema}')")

        uses_timestamp = schema.get("use_timestamp", False)
        uses_version = schema.get("use_version", False)
        uses_stepping = schema.get("use_stepping", False)

        if uses_version:
            payload = payload | {"version": self._version_id}
        if uses_timestamp:
            payload = payload | {"timestamp": datetime.now()}
        if uses_stepping:
            if "stepping" not in payload:
                raise KeyError("Expected 'stepping' field in payload")

        obj = conn.execute(
            sqla.dialects.sqlite.insert(table).on_conflict_do_nothing(), payload
        )

        serial = obj.lastrowid
        if serial is not None:
            return serial

        cls_name = schema["name"]
        raise RuntimeError(
            f"Insert error when creating new entry for storable class '{cls_name}' (payload={payload})"
        )
