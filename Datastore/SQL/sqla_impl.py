import functools
import random
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Union, Mapping, Callable, Optional

import ray
import sqlalchemy as sqla
from ray.actor import ActorHandle
from sqlalchemy.exc import IntegrityError

from CosmologyConcepts import wavenumber, wavenumber_exit_time
from Datastore.SQL.ObjectFactories.LambdaCDM import sqla_LambdaCDM_factory
from Datastore.SQL.ObjectFactories.MatterTransferFunction import (
    sqla_MatterTransferFunctionIntegration_factory,
    sqla_MatterTransferFunctionValue_factory,
    sqla_MatterTransferFunctionTagAssociation_factory,
)
from Datastore.SQL.ObjectFactories.TensorGreenFunction import (
    sqla_TensorGreenFunctionIntegration_factory,
    sqla_TensorGreenFunctionValue_factory,
    sqla_TensorGreenFunctionTagAssociation_factory,
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
from MetadataConcepts import version
from Units.base import UnitsLike
from defaults import DEFAULT_STRING_LENGTH
from utilities import WallclockTimer

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
    "MatterTransferFunctionIntegration": sqla_MatterTransferFunctionIntegration_factory,
    "MatterTransferFunctionIntegration_tags": sqla_MatterTransferFunctionTagAssociation_factory,
    "MatterTransferFunctionValue": sqla_MatterTransferFunctionValue_factory,
    "TensorGreenFunctionIntegration": sqla_TensorGreenFunctionIntegration_factory,
    "TensorGreenFunctionIntegration_tags": sqla_TensorGreenFunctionTagAssociation_factory,
    "TensorGreenFunctionValue": sqla_TensorGreenFunctionValue_factory,
}

_replicate_tables = [
    "version",
    "store_tag",
    "redshift",
    "wavenumber",
    "tolerance",
    "LambdaCDM",
    "IntegrationSolver",
]

_shard_tables = {
    "wavenumber_exit_time": "k",
    "MatterTransferFunctionIntegration": "k",
    "MatterTransferFunctionValue": "k",
    "TensorGreenFunctionIntegration": "k",
    "TensorGreenFunctionValue": "k",
}

_FactoryMappingType = Mapping[str, SQLAFactoryBase]
_TableMappingType = Mapping[str, sqla.Table]
_InserterMappingType = Mapping[str, Callable]

_TableSerialMappingType = Mapping[str, int]


@ray.remote
class StoreIdBroker:
    def __init__(self):
        self._tables: _TableSerialMappingType = {}

    def notify_largest_store_ids(self, tables: _TableSerialMappingType):
        for table, max_serial in tables.items():
            if table not in self._tables:
                self._tables[table] = 1
            else:
                self._tables[table] = max(max_serial + 1, self._tables[table])

    def next_serial(self, table: str):
        if table in self._tables:
            return self._tables[table]

        self._tables[table] = 1
        return 1

    def increment_serial(self, table: str):
        if table not in self._tables:
            raise RuntimeError("increment_serial() called on non-existent table")

        self._tables[table] += 1


@ray.remote
class Datastore:
    def __init__(
        self,
        version_label: str,
        db_name: PathType,
        version_serial: int = None,
        timeout: int = None,
        my_name: str = None,
        serial_broker: Optional[ActorHandle] = None,
    ):
        """
        Initialize an SQL datastore object
        :param version_id: version identifier used to tag results written in to the store
        """
        self._timeout = timeout
        self._my_name = my_name
        self._serial_broker = serial_broker

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
                f'Specified database file "{str(self._db_file)}" is a directory'
            )
        elif not self._db_file.exists():
            # create parent directories if they do not already exist
            self._db_file.parents[0].mkdir(exist_ok=True, parents=True)
            self._create_engine()

            # generate tables defined by any registered storable classes
            self._build_storage_schema()
            self._create_storage_tables()
        else:
            self._create_engine()
            self._build_storage_schema()
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

    def _build_storage_schema(self):
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
                        "version", sqla.Integer, sqla.ForeignKey("version.serial")
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

    def _create_storage_tables(self):
        for tab in self._tables.values():
            tab.create(self._engine)

    def _ensure_registered_schema(self, cls_name: str):
        if cls_name not in self._factories:
            raise RuntimeError(
                f'No storable class of type "{cls_name}" has been registered'
            )

    def _validate_on_startup(self):
        printed_header = False

        for cls_name, record in self._schema.items():
            if record["validate_on_startup"]:
                factory = self._factories[cls_name]

                tab = record["table"]

                with self._engine.begin() as conn:
                    msgs = factory.validate_on_startup(conn, tab, self._tables)

                if len(msgs) == 0:
                    continue

                if not printed_header:
                    if self._my_name is not None:
                        print(
                            f'!! INTEGRITY WARNING: datastore "{self._my_name}" (physical file {str(self._db_file)})'
                        )
                        printed_header = True

                for line in msgs:
                    print(line)

    def object_get(self, ObjectClass, **kwargs):
        with WallclockTimer() as timer:
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

                # payload_size = len(payload_data)
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
        #         f'** Datastore.object_get() finished in time {timer.elapsed:.3g} sec for size={payload_size} object group of class "{cls_name}" = {float(timer.elapsed)/payload_size:.3g} sec per item'
        #     )

        if scalar:
            return objects[0]

        return objects

    def object_store(self, objects):
        with WallclockTimer() as timer:
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
                    cls_name = type(obj).__name__
                    self._ensure_registered_schema(cls_name)

                    with WallclockTimer() as store_timer:
                        # print(f'**   starting store for object of type "{cls_name}"')

                        self._ensure_registered_schema(cls_name)
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

        # print(f"** Datastore.object_store() finished in time {timer.elapsed:.3g} sec")

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
        increment_serial = False
        if not uses_serial:
            if "serial" in payload:
                del payload["serial"]

        else:
            # if a serial number has already been provided, then assume we are running as a replica/shard and
            # have been provided with a correct serial number. Otherwise, obtain one from the broker, if one is in use.
            # Otherwise, assume the database engine will assign the next available serial
            # We shouldn't do this with the "version", however, which has to be treated specially
            if (
                "serial" not in payload
                and cls_name != "version"
                and self._serial_broker is not None
            ):
                payload["serial"] = ray.get(
                    self._serial_broker.next_serial.remote(cls_name),
                )
                increment_serial = True

        if uses_version:
            payload = payload | {"version": self._version.store_id}
        if uses_timestamp:
            payload = payload | {"timestamp": datetime.now()}
        if uses_stepping:
            if "stepping" not in payload:
                raise KeyError("Expected 'stepping' field in payload")

        obj = conn.execute(sqla.insert(table), payload)

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

        # update serial number information held by the broker, if needed, and if one is in use
        if increment_serial:
            ray.get(self._serial_broker.increment_serial.remote(cls_name))

        if uses_serial:
            return reported_serial

    def object_validate(self, objects):
        with WallclockTimer() as timer:
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
                    self._ensure_registered_schema(cls_name)

                    with WallclockTimer() as store_timer:
                        self._ensure_registered_schema(cls_name)
                        record = self._schema[cls_name]

                        tab = record["table"]

                        factory = self._factories[cls_name]

                        output_flags.append(
                            factory.validate(
                                objects,
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
        values = {}

        with self._engine.begin() as conn:
            for name, schema in self._schema.items():
                if schema.get("uses_serial", False):
                    table = schema["table"]
                    largest_serial = conn.execute(
                        sqla.select(
                            sqla.func.max(table.c.serial),
                        )
                    ).scalar()
                    values[name] = largest_serial

        return values


class ShardedPool:
    """
    ShardedPool manages a pool of datastore actors that cooperate to
    manage a sharded SQL database
    """

    def __init__(self, version_label: str, db_name: PathType, timeout=None, shards=10):
        """
        Initialize a pool of datastore actors
        :param version_label:
        """
        self._version_label = version_label

        self._db_name = db_name
        self._timeout = timeout
        self._shards = max(shards, 1)

        # resolve concerts the supplied db_name to an absolute path, resolving symlinks if necessary
        # this database file will be taken to be the primary database
        self._primary_file = Path(db_name).resolve()

        self._shard_db_files = {}
        self._wavenumber_keys = {}

        self._broker = StoreIdBroker.options(name="StoreIdBroker").remote()

        # if primary file is absent, all shard databases should be likewise absent
        if self._primary_file.is_dir():
            raise RuntimeError(
                f'Specified database file "{str(self._db_file)}" is a directory'
            )
        if not self._primary_file.exists():
            # ensure parent directories also exist
            self._primary_file.parents[0].mkdir(exist_ok=True, parents=True)

            stem = self._primary_file.stem
            for i in range(self._shards):
                shard_stem = f"{stem}-shard{i:04d}"
                shard_file = self._primary_file.with_stem(shard_stem)

                if shard_file.exists():
                    raise RuntimeError(
                        f'Primary database is missing, but shard "{str(shard_file)}" already exists'
                    )

                self._shard_db_files[i] = shard_file

            self._create_engine()
            self._write_shard_data()

            print(
                f'>> Created sharded datastore "{str(self._primary_file)}" with {self._shards} shards'
            )

        # otherwise, if primary exists, try to read in shard configuration from it.
        # Then, all shard databases must be present
        else:
            self._create_engine()
            self._read_shard_data()

            num_shards = len(self._shard_db_files)
            print(
                f'>> Opened existing sharded datastore "{str(self._primary_file)}" with {num_shards} shards'
            )

            if num_shards == 0:
                raise RuntimeError(
                    "No shard records were read from the sharded datastore"
                )
            if num_shards != self._shards:
                print(
                    f"!! WARNING: number of shards read from database (={num_shards}) does not match specified number of shards (={self._shards})"
                )

        # create actor pool of datastores, one for each shard
        # we read the version serial number from the first shard that we create
        shard_ids = list(self._shard_db_files.keys())

        shard0_key = shard_ids.pop()
        shard0_file = self._shard_db_files[shard0_key]

        # create the first shard datastore
        shard0_store = Datastore.options(name=f"shard{shard0_key:04d}-store").remote(
            version_label=version_label,
            db_name=shard0_file,
            timeout=self._timeout,
            my_name=f"shard{shard0_key:04d}-store",
            serial_broker=self._broker,
        )
        self._shards = {shard0_key: shard0_store}

        # get the version label from this store
        self._version = ray.get(
            shard0_store.object_get.remote(version, label=version_label)
        )

        # populate the remaining pool of shard stores
        self._shards.update(
            {
                key: Datastore.options(name=f"shard{key:04d}-store").remote(
                    version_label=version_label,
                    version_serial=self._version.store_id,
                    db_name=self._shard_db_files[key],
                    timeout=self._timeout,
                    my_name=f"shard{key:04d}-store",
                    serial_broker=self._broker,
                )
                for key in shard_ids
            }
        )

        # query a list of largest serial numbers from each shard, and notify these to the broker actor
        max_serial_data = ray.get(
            [shard.read_largest_store_ids.remote() for shard in self._shards.values()]
        )
        ray.get(
            [
                self._broker.notify_largest_store_ids.remote(payload)
                for payload in max_serial_data
            ]
        )

    def _create_engine(self):
        connect_args = {}
        if self._timeout is not None:
            connect_args["timeout"] = self._timeout

        self._engine = sqla.create_engine(
            f"sqlite:///{self._db_name}",
            future=True,
            connect_args=connect_args,
        )
        self._metadata = sqla.MetaData()

        self._shard_file_table = sqla.Table(
            "shards",
            self._metadata,
            sqla.Column("serial", sqla.Integer, primary_key=True, nullable=False),
            sqla.Column("filename", sqla.String(DEFAULT_STRING_LENGTH), nullable=False),
        )
        self._shard_key_table = sqla.Table(
            "shard_keys",
            self._metadata,
            sqla.Column(
                "wavenumber_serial", sqla.Integer, primary_key=True, nullable=False
            ),
            sqla.Column(
                "shard_id",
                sqla.Integer,
                sqla.ForeignKey("shards.serial"),
                nullable=False,
            ),
        )

    def _write_shard_data(self):
        self._shard_file_table.create(self._engine)
        self._shard_key_table.create(self._engine)

        with self._engine.begin() as conn:
            values = [
                {"serial": key, "filename": str(db_name)}
                for key, db_name in self._shard_db_files.items()
            ]

            conn.execute(sqla.insert(self._shard_file_table), values)
            conn.commit()

    def _read_shard_data(self):
        with self._engine.begin() as conn:
            rows = conn.execute(
                sqla.select(
                    self._shard_file_table.c.serial,
                    self._shard_file_table.c.filename,
                )
            )

            for row in rows:
                serial = row.serial
                filename = Path(row.filename)

                if serial in self._shard_db_files:
                    raise RuntimeError(
                        f'Shard #{serial} already exists (database file="{str(filename)}", existing file="{str(self._shard_db_files[serial])}")'
                    )

                self._shard_db_files[row.serial] = Path(row.filename)

            keys = conn.execute(
                sqla.select(
                    self._shard_key_table.c.wavenumber_serial,
                    self._shard_key_table.c.shard_id,
                )
            )

            for key in keys:
                self._wavenumber_keys[key.wavenumber_serial] = key.shard_id

    def object_get(self, ObjectClass, **kwargs):
        if isinstance(ObjectClass, str):
            cls_name = ObjectClass
        else:
            cls_name = ObjectClass.__name__

        if cls_name in _replicate_tables:
            return self._get_impl_replicated_table(cls_name, kwargs)

        if cls_name in _shard_tables.keys():
            return self._get_impl_sharded_table(cls_name, kwargs)

        raise RuntimeError(
            f'Unable to dispatch object_get() for item of type "{cls_name}"'
        )

    def _get_impl_replicated_table(self, cls_name, kwargs):
        # pick a shard id at random to be the "controlling" shard
        shard_ids = list(self._shards.keys())
        i = random.randrange(len(shard_ids))

        # swap this entry with the last element, then pop it
        shard_ids[i], shard_ids[-1] = shard_ids[-1], shard_ids[i]
        shard_key = shard_ids.pop()

        # for replicated tables, we should query/insert into *one* datastore, and then enforce
        # that all other datastores get the same store_id; here, there is no need to use our internal
        # information about the next-allocated store_id, and in fact doing so would make the logic
        # here much more complicated. So we avoid that.
        ref = self._shards[shard_key].object_get.remote(cls_name, **kwargs)
        objects = ray.get(ref)

        if "payload_data" in kwargs:
            payload_data = kwargs["payload_data"]

            if len(payload_data) != len(objects):
                raise RuntimeError(
                    f"object_get() data returned from selected datastore (shared={shard_key}) has a different length (length={len(objects)}) to payload data (length={len(payload_data)})"
                )

            # add explicit serial specifier
            new_payload = []
            for i in range(len(payload_data)):
                if not hasattr(objects[i], "_deserialized"):
                    payload_data[i]["serial"] = objects[i].store_id
                    new_payload.append(payload_data[i])

            # queue work items to replicate each object in all other shards (recall that shard_key has already been popped from shard_ids,
            # so there is no double insertion here)
            ray.get(
                [
                    self._shards[key].object_get.remote(
                        cls_name, payload_data=new_payload
                    )
                    for key in shard_ids
                ]
            )
        else:
            # this was a scalar get
            if not hasattr(objects, "_deserialized"):
                ray.get(
                    [
                        self._shards[key].object_get.remote(
                            cls_name, serial=objects.store_id, **kwargs
                        )
                        for key in shard_ids
                    ]
                )

        # test whether this query was for a shard key, and, if so, assign any shard keys
        # that are missing
        if cls_name == "wavenumber":
            self._assign_shard_keys(objects)

        # return original object (we just discard any copies returned from other shards)
        return ref

    def _get_impl_sharded_table(self, cls_name, kwargs):
        # for sharded tables, we should query/insert into only the appropriate shard
        shard_key_field = _shard_tables[cls_name]

        # determine which shard contains this item
        if "payload_data" in kwargs:
            payload_data = kwargs["payload_data"]

            work_refs = []
            for item in payload_data:
                k = item[shard_key_field]
                shard_id = self._wavenumber_keys[self._get_k_store_id(k)]

                work_refs.append(
                    self._shards[shard_id].object_get.remote(cls_name, **item)
                )
                return work_refs
            # TODO: consider consolidating all objects for the same shard into a list, for efficiency
        else:
            k = kwargs[shard_key_field]
            shard_id = self._wavenumber_keys[self._get_k_store_id(k)]

            return self._shards[shard_id].object_get.remote(cls_name, **kwargs)

    def object_store(self, objects):
        # we only expect to call object_store on sharded objects
        if isinstance(objects, list) or isinstance(objects, tuple):
            payload_data = objects
            scalar = False
        else:
            payload_data = [objects]
            scalar = True

        work_refs = []
        for item in payload_data:
            cls_name = type(item).__name__

            if cls_name not in _shard_tables.keys():
                raise RuntimeError(
                    f'Unable to dispatch object_store() for item of type "{cls_name}"'
                )

            if not hasattr(item, "k"):
                raise RuntimeError(
                    f'Unable to determine shard, because object of type "{cls_name}" has no "k" attribute'
                )

            k = item.k
            shard_id = self._wavenumber_keys[self._get_k_store_id(k)]

            work_refs.append(self._shards[shard_id].object_store.remote(item))
            # TODO: consider consolidating all objects for the same shard into a list, for efficiency

        if scalar:
            return work_refs[0]

        return work_refs

    def object_validate(self, objects):
        # we only expect to call object_store on sharded objects
        if isinstance(objects, list) or isinstance(objects, tuple):
            payload_data = objects
            scalar = False
        else:
            payload_data = [objects]
            scalar = True

        work_refs = []
        for item in payload_data:
            cls_name = type(item).__name__

            if cls_name not in _shard_tables.keys():
                raise RuntimeError(
                    f'Unable to dispatch object_store() for item of type "{cls_name}"'
                )

            if not hasattr(item, "k"):
                raise RuntimeError(
                    f'Unable to determine shard, because object of type "{cls_name}" has no "k" attribute'
                )

            k = item.k
            shard_id = self._wavenumber_keys[self._get_k_store_id(k)]

            work_refs.append(self._shards[shard_id].object_validate.remote(item))
            # TODO: consider consolidating all objects for the same shard into a list, for efficiency

        if scalar:
            return work_refs[0]

        return work_refs

    def _get_k_store_id(self, obj):
        if isinstance(obj, wavenumber):
            return obj.store_id

        if isinstance(obj, wavenumber_exit_time):
            return obj.k.store_id

        raise RuntimeError(
            f'Could not determine shard index k for obejct of type "{type(obj)}"'
        )

    def _assign_shard_keys(self, obj):
        if isinstance(obj, list):
            data = obj
        else:
            data = [obj]

        # assign any shard keys that we can, without going out to the database
        # (because this is bound to be slower)
        missing_keys = []
        for item in data:
            if not isinstance(item, wavenumber):
                raise RuntimeError("shard keys should be of type wavenumber")

            if item.store_id not in self._wavenumber_keys:
                missing_keys.append(item)

        # if no work to do, return
        if len(missing_keys) == 0:
            return

        # otherwise, we have to populate keys
        # try to load balance by working out which shard has fewest wavenumbers
        loads = {key: 0 for key in self._shards.keys()}
        for shard in self._wavenumber_keys.values():
            loads[shard] = loads[shard] + 1

        with self._engine.begin() as conn:
            for item in missing_keys:
                # find which shard has the current minimum load
                if len(loads) > 0:
                    new_shard = min(loads, key=loads.get)
                else:
                    new_shard = list(self._shards.keys()).pop()

                # insert a new record for this key
                conn.execute(
                    sqla.insert(self._shard_key_table),
                    {"wavenumber_id": item.store_id, "shard_id": new_shard},
                )

                self._wavenumber_keys[item.store_id] = new_shard
                loads[new_shard] = loads[new_shard] + 1

                # print(
                #     f">> assigned shard #{new_shard} to wavenumber object #{item.store_id} (k={item.k_inv_Mpc}/Mpc)"
                # )

            conn.commit()

    def read_wavenumber_table(self, units: UnitsLike):
        """
        Read the wavenumber value table from one of the database shards
        :param units:
        :return:
        """
        # we only need to read the wavenumber table from a single shard, so pick one at random
        shard_ids = list(self._shards.keys())
        i = random.randrange(len(shard_ids))

        # swap this entry with the last element, then pop it
        shard_ids[i], shard_ids[-1] = shard_ids[-1], shard_ids[i]
        shard_key = shard_ids.pop()

        return self._shards[shard_key].read_wavenumber_table.remote(units=units)

    def read_redshift_table(self):
        """
        Read the redshift value table from one of the database shards
        :return:
        """
        # we only need to read the redshift table from a single shard, so pick one at random
        shard_ids = list(self._shards.keys())
        i = random.randrange(len(shard_ids))

        # swap this entry with the last element, then pop it
        shard_ids[i], shard_ids[-1] = shard_ids[-1], shard_ids[i]
        shard_key = shard_ids.pop()

        return self._shards[shard_key].read_redshift_table.remote()
