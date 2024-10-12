import functools
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Union, Mapping, Callable, Optional, List

import ray
import sqlalchemy as sqla
from ray.actor import ActorHandle
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from Datastore.SQL.ClientPool import SerialPoolManager, SerialLeaseManager
from Datastore.SQL.ObjectFactories.BackgroundModel import (
    sqla_BackgroundModelFactory,
    sqla_BackgroundModelTagAssociation_factory,
    sqla_BackgroundModelValue_factory,
)
from Datastore.SQL.ObjectFactories.GkNumericalIntegration import (
    sqla_GkNumericalIntegration_factory,
    sqla_GkNumericalValue_factory,
    sqla_GkNumericalTagAssociation_factory,
)
from Datastore.SQL.ObjectFactories.GkSource import (
    sqla_GkSourceTagAssociation_factory,
    sqla_GkSource_factory,
    sqla_GkSourceValue_factory,
)
from Datastore.SQL.ObjectFactories.GkWKBIntegration import (
    sqla_GkWKBIntegration_factory,
    sqla_GkWKBTagAssociation_factory,
    sqla_GkWKBValue_factory,
)
from Datastore.SQL.ObjectFactories.LambdaCDM import sqla_LambdaCDM_factory
from Datastore.SQL.ObjectFactories.TensorSource import (
    sqla_TensorSource_factory,
    sqla_TensorSourceTagAssocation_factory,
    sqla_TensorSourceValue_factory,
)
from Datastore.SQL.ObjectFactories.TkNumericalIntegration import (
    sqla_TkNumericalIntegration_factory,
    sqla_TkNumericalValue_factory,
    sqla_TkNumericalTagAssociation_factory,
)
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from Datastore.SQL.ObjectFactories.integration_metadata import (
    sqla_IntegrationSolver_factory,
)
from Datastore.SQL.ObjectFactories.redshift import sqla_redshift_factory
from Datastore.SQL.ObjectFactories.store_tag import sqla_store_tag_factory
from Datastore.SQL.ObjectFactories.tolerance import sqla_tolerance_factory
from Datastore.SQL.ObjectFactories.version import sqla_version_factory
from Datastore.SQL.ObjectFactories.wavenumber import (
    sqla_wavenumber_factory,
    sqla_wavenumber_exit_time_factory,
)
from Datastore.SQL.ProfileAgent import ProfileBatcher, ProfileBatchManager
from MetadataConcepts import version

VERSION_ID_LENGTH = 64


PathType = Union[str, PathLike]


_factories = {
    "version": sqla_version_factory,
    "store_tag": sqla_store_tag_factory,
    "redshift": sqla_redshift_factory,
    "wavenumber": sqla_wavenumber_factory,
    "wavenumber_exit_time": sqla_wavenumber_exit_time_factory,
    "tolerance": sqla_tolerance_factory,
    "LambdaCDM": sqla_LambdaCDM_factory,
    "IntegrationSolver": sqla_IntegrationSolver_factory,
    "BackgroundModel": sqla_BackgroundModelFactory,
    "BackgroundModel_tags": sqla_BackgroundModelTagAssociation_factory,
    "BackgroundModelValue": sqla_BackgroundModelValue_factory,
    "TkNumericalIntegration": sqla_TkNumericalIntegration_factory,
    "TkNumerical_tags": sqla_TkNumericalTagAssociation_factory,
    "TkNumericalValue": sqla_TkNumericalValue_factory,
    "GkNumericalIntegration": sqla_GkNumericalIntegration_factory,
    "GkNumerical_tags": sqla_GkNumericalTagAssociation_factory,
    "GkNumericalValue": sqla_GkNumericalValue_factory,
    "TensorSource": sqla_TensorSource_factory,
    "TensorSource_tags": sqla_TensorSourceTagAssocation_factory,
    "TensorSourceValue": sqla_TensorSourceValue_factory,
    "GkWKBIntegration": sqla_GkWKBIntegration_factory,
    "GkWKB_tags": sqla_GkWKBTagAssociation_factory,
    "GkWKBValue": sqla_GkWKBValue_factory,
    "GkSource": sqla_GkSource_factory,
    "GkSource_tags": sqla_GkSourceTagAssociation_factory,
    "GkSourceValue": sqla_GkSourceValue_factory,
}

_FactoryMappingType = Mapping[str, SQLAFactoryBase]
_TableMappingType = Mapping[str, sqla.Table]
_InserterMappingType = Mapping[str, Callable]

_drop_actions = {
    "numerical": ["GkNumericalIntegration", "GkNumerical_tags", "GkNumericalValue"],
    "wkb": ["GkWKBIntegration", "GkWKB_tags", "GkWKBValue"],
    "source": ["GkSource", "GkSource_tags", "GkSourceValue"],
}


@ray.remote
class Datastore:
    def __init__(
        self,
        version_label: str,
        db_name: PathType,
        version_serial: Optional[int] = None,
        timeout: Optional[int] = None,
        my_name: Optional[str] = None,
        serial_broker: Optional[ActorHandle] = None,
        profile_agent: Optional[ActorHandle] = None,
        prune_unvalidated: Optional[bool] = False,
        drop_actions: Optional[List[str]] = None,
    ):
        """
        Initialize an SQL datastore object
        """
        self._timeout = timeout
        self._my_name = my_name

        self._prune_unvalidated = prune_unvalidated

        profile_label = my_name if my_name else "anonymous-Datastore"
        self._profile_batcher = ProfileBatcher(profile_agent, profile_label)

        self._serial_broker = serial_broker
        self._serial_manager = SerialPoolManager(
            profiler=self._profile_batcher, broker=self._serial_broker
        )

        self._db_file = Path(db_name).resolve()

        # initialize set of registered storable class adapters
        self._factories: _FactoryMappingType = {}
        self.register_factories(_factories)

        # initialize empty dict of storage schema
        # each record collects SQLAlchemy column and table definitions, queries, etc., for a registered storable class factories
        self._tables: _TableMappingType = {}
        self._inserters: _InserterMappingType = {}
        self._schema = {}

        if self._db_file.is_dir():
            raise RuntimeError(
                f'Specified datastore database file "{str(self._db_file)}" is a directory'
            )
        elif not self._db_file.exists():
            # create parent directories if they do not already exist
            self._db_file.parents[0].mkdir(exist_ok=True, parents=True)
            self._create_engine()
            self._build_schema()
            self._ensure_tables()
        else:
            self._create_engine()
            self._build_schema()
            self._drop_actions(drop_actions)
            self._ensure_tables()
            self._validate_on_startup()

        # convert version label to a version object
        # if a serial is specified, we are probably running as a replica, and we need to ensure that the specified
        # serial number is honoured in order to ensure integrity across multiple shards
        version_payload = {"label": version_label}
        if version_serial is not None:
            version_payload["serial"] = version_serial

        self._version = self.object_get(version, **version_payload)

        if version_serial is not None and self._version.store_id != version_serial:
            raise IntegrityError(
                f"Serial number of version label (={self._version.store_id}) does not match specified value (={version_serial})"
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # clean up SQLAlchemy engine if it exists
        if self._engine is not None:
            self._engine.dispose()

        # clean up the various services that we use
        if self._serial_manager is not None:
            self._serial_manager.clean_up()

        if self._profile_batcher is not None:
            self._profile_batcher.clean_up()

    def _create_engine(self):
        """
        Create and initialize an SQLAlchemy engine corresponding to the name data container,
        :return:
        """
        connect_args = {}
        if self._timeout is not None:
            connect_args["timeout"] = self._timeout

        self._engine = sqla.create_engine(
            f"sqlite:///{self._db_file}",
            future=True,
            connect_args=connect_args,
        )
        self._metadata = sqla.MetaData()
        self._inspector = sqla.inspect(self._engine)

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
            # print(f"Registered storable class factory '{cls_name}'")

    def _build_schema(self):
        # iterate through all registered storage adapters, querying them for the columns
        # they need to persist their data
        for cls_name, factory in self._factories.items():
            if cls_name in self._schema:
                raise RuntimeWarning(
                    f"Duplicate registered factory for storable class '{cls_name}'"
                )

            # query class for a list of columns that it wants to store
            registration_data = factory.register()

            schema = {
                "name": cls_name,
                "validate_on_startup": registration_data.get(
                    "validate_on_startup", False
                ),
            }

            # does this storage object require its own table?
            if registration_data is not None:
                # generate main table for this adapter class
                tab = sqla.Table(
                    cls_name,
                    self._metadata,
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

                    _stepping_mode = (
                        None if not isinstance(use_stepping, str) else use_stepping
                    )
                    schema["stepping_mode"] = _stepping_mode

                # append all columns supplied by the class
                sqla_columns = registration_data.get("columns", [])
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

                # print(
                #     f"Registered storage schema for storable class adapter '{cls_name}' with database table '{tab.name}'"
                # )
            else:
                schema["table"] = None
                schema["insert"] = None

                # print(
                #     f"Registered storage schema for storable class adapter '{cls_name}' without database table"
                # )

            self._schema[cls_name] = schema

    def _ensure_tables(self):
        for name, tab in self._tables.items():
            if not self._inspector.has_table(name):
                tab.create(self._engine)

    def _ensure_registered_schema(self, cls_name: str):
        if cls_name not in self._factories:
            raise RuntimeError(
                f'No storable class of type "{cls_name}" has been registered'
            )

    def _drop_actions(self, actions):
        for action in actions:
            if action in _drop_actions:
                print(f'Datastore: dropping tables for "{action}"')
                drop_list = _drop_actions[action]
                for table_name in drop_list:
                    if self._inspector.has_table(table_name):
                        tab = self._tables.get(table_name, None)
                        if tab is not None:
                            tab.drop(self._engine)
                            self._metadata.remove(table_name)
            else:
                print(f'Datastore: unknown drop action "{action}"')

    def _validate_on_startup(self):
        printed_header = False

        for cls_name, record in self._schema.items():
            if record["validate_on_startup"]:
                factory = self._factories[cls_name]

                tab = record["table"]

                with self._engine.begin() as conn:
                    msgs = factory.validate_on_startup(
                        conn, tab, self._tables, self._prune_unvalidated
                    )

                if len(msgs) == 0:
                    continue

                if not printed_header:
                    if self._my_name is not None:
                        print(
                            f'!! INTEGRITY WARNING ({datetime.now().replace(microsecond=0).isoformat()}): datastore "{self._my_name}" (physical file {str(self._db_file)})'
                        )
                        printed_header = True

                for line in msgs:
                    print(line)

    def object_get(self, ObjectClass, **kwargs):
        if isinstance(ObjectClass, str):
            cls_name = ObjectClass
        else:
            cls_name = ObjectClass.__name__

        profile_metadata = {"object": cls_name}
        if "payload_data" in kwargs:
            profile_metadata.update(
                {"type": "vector", "size": len(kwargs["payload_data"])}
            )
        else:
            profile_metadata.update({"type": "scalar"})

        with ProfileBatchManager(
            self._profile_batcher, "object_get", profile_metadata
        ) as mgr:
            self._ensure_registered_schema(cls_name)
            record = self._schema[cls_name]

            tab = record["table"]
            inserter = record["insert"]

            # obtain type of factory class for this storable
            factory = self._factories[cls_name]

            if "payload_data" in kwargs:
                payload_data = kwargs["payload_data"]
                scalar = False
            else:
                payload_data = [kwargs]
                scalar = True

            try:
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
            except SQLAlchemyError as e:
                print(
                    f"!! Database error in datastore build() [store={self._my_name}, physical store={self._db_file}]"
                )
                print(f"|  payload data = {payload_data}")
                print(f"|  {e}")
                raise e

        if scalar:
            return objects[0]

        return objects

    def object_read_batch(self, ObjectClass, **payload):
        if isinstance(ObjectClass, str):
            cls_name = ObjectClass
        else:
            cls_name = ObjectClass.__name__

        with ProfileBatchManager(
            self._profile_batcher, "object_read_batch", {"object": cls_name}
        ) as mgr:
            self._ensure_registered_schema(cls_name)
            record = self._schema[cls_name]

            tab = record["table"]
            factory = self._factories[cls_name]

            try:
                with self._engine.begin() as conn:
                    objects = factory.read_batch(
                        payload=payload, conn=conn, table=tab, tables=self._tables
                    )
                    mgr.update_metadata({"read_objects": len(objects)})

            except SQLAlchemyError as e:
                print(
                    f"!! Database error in datastore build() [store={self._my_name}, physical store={self._db_file}]"
                )
                print(f"|  payload data = {payload}")
                print(f"|  {e}")
                raise e

        return objects

    def object_store(self, objects):
        if isinstance(objects, list) or isinstance(objects, tuple):
            payload_data = objects
            scalar = False
        else:
            payload_data = [objects]
            scalar = True

        output_objects = []
        with self._engine.begin() as conn:
            for obj in payload_data:
                cls_name = type(obj).__name__
                with ProfileBatchManager(
                    self._profile_batcher, "object_store_item", {"object": cls_name}
                ) as mgr:
                    self._ensure_registered_schema(cls_name)
                    record = self._schema[cls_name]

                    tab = record["table"]
                    inserter = record["insert"]

                    factory = self._factories[cls_name]

                    output_objects.append(
                        factory.store(
                            obj,
                            conn=conn,
                            table=tab,
                            inserter=inserter,
                            tables=self._tables,
                            inserters=self._inserters,
                        )
                    )

            conn.commit()

        if scalar:
            return output_objects[0]

        return output_objects

    def _insert(self, schema, table, conn, payload):
        if table is None:
            raise RuntimeError(f"Attempt to insert into null table (schema='{schema}')")

        uses_serial = schema.get("use_serial", True)
        uses_timestamp = schema.get("use_timestamp", False)
        uses_version = schema.get("use_version", False)
        uses_stepping = schema.get("use_stepping", False)

        cls_name = schema["name"]

        # remove any "serial" field from the payload, if this table does not use serial numbers
        if not uses_serial:
            if "serial" in payload:
                del payload["serial"]

        # if a serial number has already been provided, then assume we are running as a replica/shard and
        # have been provided with a correct serial number. Otherwise, obtain one from the broker, if one is in use.
        # Otherwise, assume the database engine will assign the next available serial
        # We shouldn't do this with the "version", however, which has to be treated specially
        with SerialLeaseManager(
            self._serial_manager,
            cls_name,
            uses_serial and ("serial" not in payload) and (cls_name != "version"),
        ) as mgr:
            commit_serial = False
            if mgr.serial is not None:
                payload["serial"] = mgr.serial
                commit_serial = True

            if uses_version:
                payload = payload | {"version": self._version.store_id}
            if uses_timestamp:
                payload = payload | {"timestamp": datetime.now()}
            if uses_stepping:
                if "stepping" not in payload:
                    raise KeyError("Expected 'stepping' field in payload")

            obj = conn.execute(sqla.insert(table), payload)

            if commit_serial:
                mgr.commit()

            reported_serial = obj.lastrowid

            if reported_serial is None:
                raise RuntimeError(
                    f"Insert error when creating new entry for storable class '{cls_name}' (payload={payload})"
                )

            expected_serial = payload.get("serial", None)
            if (
                "serial" in payload
                and expected_serial is not None
                and reported_serial != expected_serial
            ):
                raise RuntimeError(
                    f"Inserted store_id reported from database engine (={reported_serial}) does not agree with supplied store_id (={expected_serial}"
                )

        if uses_serial:
            return reported_serial

    def object_validate(self, objects):
        if isinstance(objects, list) or isinstance(objects, tuple):
            payload_data = objects
            scalar = False
        else:
            payload_data = [objects]
            scalar = True

        output_flags = []
        with self._engine.begin() as conn:
            for obj in payload_data:
                cls_name = type(obj).__name__
                with ProfileBatchManager(
                    self._profile_batcher, "object_validate_item", {"object": cls_name}
                ) as mgr:
                    self._ensure_registered_schema(cls_name)
                    record = self._schema[cls_name]

                    tab = record["table"]

                    factory = self._factories[cls_name]

                    output_flags.append(
                        factory.validate(
                            obj,
                            conn=conn,
                            table=tab,
                            tables=self._tables,
                        )
                    )

            conn.commit()

        if scalar:
            return output_flags[0]

        return output_flags

    def read_wavenumber_table(self, units):
        """
        Read the wavenumber value table from the database
        :param units:
        :return:
        """
        with ProfileBatchManager(self._profile_batcher, "read_wavenumber_table") as mgr:
            self._ensure_registered_schema("wavenumber")
            record = self._schema["wavenumber"]

            tab = record["table"]
            factory = self._factories["wavenumber"]

            with self._engine.begin() as conn:
                objects = factory.read_table(conn, tab, units)

        return objects

    def read_redshift_table(self):
        """
        Read the redshift value table from the database
        :return:
        """
        with ProfileBatchManager(self._profile_batcher, "read_redshift_table") as mgr:
            self._ensure_registered_schema("redshift")
            record = self._schema["redshift"]

            tab = record["table"]
            factory = self._factories["redshift"]

            with self._engine.begin() as conn:
                objects = factory.read_table(conn, tab)

        return objects

    def read_largest_store_ids(self):
        """
        Iterate through all registered tables, and determine the largest serial value we are holding.
        This is mostly useful to ShardedPool, which uses this API to determine which store_id values it should allocate
        to newly serialized objects
        :return:
        """
        with ProfileBatchManager(
            self._profile_batcher, "read_largest_store_ids"
        ) as mgr:
            values = {}

            with self._engine.begin() as conn:
                for name, schema in self._schema.items():
                    if schema.get("use_serial", False):
                        table = schema["table"]
                        largest_serial = conn.execute(
                            sqla.select(
                                sqla.func.max(table.c.serial),
                            )
                        ).scalar()
                        values[name] = largest_serial

            return values
