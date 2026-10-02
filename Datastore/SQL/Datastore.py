import functools
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Union, Mapping, Callable, Optional, List, Iterable, Dict, Any

import ray
import sqlalchemy as sqla
from ray.actor import ActorHandle
from sqlalchemy.exc import SQLAlchemyError

from Datastore.SQL.ClientPool import SerialPoolManager, SerialLeaseManager
from Datastore.SQL.ObjectFactories.BackgroundModel import (
    sqla_BackgroundModelFactory,
    sqla_BackgroundModelTagAssociation_factory,
    sqla_BackgroundModelValue_factory,
)
from Datastore.SQL.ObjectFactories.GkNumericIntegration import (
    sqla_GkNumericIntegration_factory,
    sqla_GkNumericValue_factory,
    sqla_GkNumericTagAssociation_factory,
)
from Datastore.SQL.ObjectFactories.GkSource import (
    sqla_GkSourceTagAssociation_factory,
    sqla_GkSource_factory,
    sqla_GkSourceValue_factory,
    sqla_GkSourceParents_factory,
)
from Datastore.SQL.ObjectFactories.GkSourcePolicy import sqla_GkSourcePolicy_factory
from Datastore.SQL.ObjectFactories.GkSourcePolicyData import (
    sqla_GkSourcePolicyData_factory,
)
from Datastore.SQL.ObjectFactories.GkWKBIntegration import (
    sqla_GkWKBIntegration_factory,
    sqla_GkWKBTagAssociation_factory,
    sqla_GkWKBValue_factory,
)
from Datastore.SQL.ObjectFactories.LambdaCDM import sqla_LambdaCDM_factory
from Datastore.SQL.ObjectFactories.OneLoopIntegral import (
    sqla_OneLoopIntegral_factory,
    sqla_OneLoopIntegralTagAssociation_factory,
)
from Datastore.SQL.ObjectFactories.QCD_Cosmology import sqla_QCDCosmology_factory
from Datastore.SQL.ObjectFactories.QuadSource import (
    sqla_QuadSource_factory,
    sqla_QuadSourceTagAssocation_factory,
    sqla_QuadSourceValue_factory,
)
from Datastore.SQL.ObjectFactories.QuadSourceIntegral import (
    sqla_QuadSourceIntegralTagAssociation_factory,
    sqla_QuadSourceIntegral_factory,
)
from Datastore.SQL.ObjectFactories.QuadSourcePolicy import sqla_QuadSourcePolicy_factory
from Datastore.SQL.ObjectFactories.TkNumericIntegration import (
    sqla_TkNumericIntegration_factory,
    sqla_TkNumericValue_factory,
    sqla_TkNumericTagAssociation_factory,
)
from Datastore.SQL.ObjectFactories.TkWKBIntegration import (
    sqla_TkWKBIntegration_factory,
    sqla_TkWKBTagAssociation_factory,
    sqla_TkWKBValue_factory,
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
from Datastore.SQL.schema import build_schema
from Datastore.replication import ReplicationMismatch
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
    "QCD_Cosmology": sqla_QCDCosmology_factory,
    "IntegrationSolver": sqla_IntegrationSolver_factory,
    "BackgroundModel": sqla_BackgroundModelFactory,
    "BackgroundModel_tags": sqla_BackgroundModelTagAssociation_factory,
    "BackgroundModelValue": sqla_BackgroundModelValue_factory,
    "TkNumericIntegration": sqla_TkNumericIntegration_factory,
    "TkNumeric_tags": sqla_TkNumericTagAssociation_factory,
    "TkNumericValue": sqla_TkNumericValue_factory,
    "TkWKBIntegration": sqla_TkWKBIntegration_factory,
    "TkWKB_tags": sqla_TkWKBTagAssociation_factory,
    "TkWKBValue": sqla_TkWKBValue_factory,
    "GkNumericIntegration": sqla_GkNumericIntegration_factory,
    "GkNumeric_tags": sqla_GkNumericTagAssociation_factory,
    "GkNumericValue": sqla_GkNumericValue_factory,
    "GkWKBIntegration": sqla_GkWKBIntegration_factory,
    "GkWKB_tags": sqla_GkWKBTagAssociation_factory,
    "GkWKBValue": sqla_GkWKBValue_factory,
    "GkSourcePolicy": sqla_GkSourcePolicy_factory,
    "GkSourcePolicyData": sqla_GkSourcePolicyData_factory,
    "GkSource": sqla_GkSource_factory,
    "GkSource_tags": sqla_GkSourceTagAssociation_factory,
    "GkSourceValue": sqla_GkSourceValue_factory,
    # the Gk rows each GkSource is assembled from (prompts/datastore-integrity, prompt 09b)
    "GkSource_parents": sqla_GkSourceParents_factory,
    "QuadSourcePolicy": sqla_QuadSourcePolicy_factory,
    "QuadSource": sqla_QuadSource_factory,
    "QuadSource_tags": sqla_QuadSourceTagAssocation_factory,
    "QuadSourceValue": sqla_QuadSourceValue_factory,
    "QuadSourceIntegral": sqla_QuadSourceIntegral_factory,
    "QuadSourceIntegral_tags": sqla_QuadSourceIntegralTagAssociation_factory,
    "OneLoopIntegral": sqla_OneLoopIntegral_factory,
    "OneLoopIntegral_tags": sqla_OneLoopIntegralTagAssociation_factory,
}

_FactoryMappingType = Mapping[str, SQLAFactoryBase]
_TableMappingType = Mapping[str, sqla.Table]
_InserterMappingType = Mapping[str, Callable]

_drop_actions = {
    "1loop-integral": [
        "OneLoopIntegral",
        "OneLoopIntegral_tags",
    ],
    "quad-source-integral": [
        "QuadSourceIntegral",
        "QuadSourceIntegral_tags",
    ],
    "gk-source-policy": ["GkSourcePolicyData"],
    "gk-source-policy-records": ["GkSourcePolicy"],
    "gk-source": ["GkSource", "GkSource_tags", "GkSourceValue", "GkSource_parents"],
    "gk-wkb": ["GkWKBIntegration", "GkWKB_tags", "GkWKBValue"],
    "gk-numeric": ["GkNumericIntegration", "GkNumeric_tags", "GkNumericValue"],
    "quad-source": ["QuadSource", "QuadSource_tags", "QuadSourceValue"],
    "tk-numeric": ["TkNumericIntegration", "TkNumeric_tags", "TkNumericValue"],
    "tk-wkb": ["TkWKBIntegration", "TkWKB_tags", "TkWKBValue"],
}
# should drop tables in a defined order, so that we do not violate foreign key constraints: an
# action whose tables declare a foreign key into another action's tables drops first. In
# particular a WKB row references the numeric row it starts from (numeric_serial,
# prompts/datastore-integrity prompt 07), so "gk-wkb" drops before "gk-numeric" and "tk-wkb" before
# "tk-numeric"; and QuadSource.Tq_serial references TkNumericIntegration (prompt 06), so
# "quad-source" drops before "tk-numeric" too
_drop_order = [
    "1loop-integral",
    "quad-source-integral",
    "gk-source-policy",
    "gk-source-policy-records",
    "gk-source",
    "gk-wkb",
    "gk-numeric",
    "quad-source",
    "tk-wkb",
    "tk-numeric",
]

# read table configuration should be a Dict with the mapping
# class_name -> {"tables_arg": bool}
ReadTableConfigType = Dict[str, Any]


@ray.remote
class Datastore:
    def __init__(
        self,
        version_label: str,
        db_name: PathType,
        replicated_tables: List[str],
        timeout: Optional[int] = None,
        my_name: Optional[str] = None,
        serial_broker: Optional[ActorHandle] = None,
        profile_agent: Optional[ActorHandle] = None,
        prune_unvalidated: Optional[bool] = False,
        drop_actions: Optional[List[str]] = None,
        read_table_config: Optional[ReadTableConfigType] = None,
    ):
        """
        Initialize an SQL datastore object.

        The constructor neither looks up nor inserts the version row
        (prompts/a3-v2-readiness, prompt 02). The version row is replicated, so ShardedPool
        finds it, or writes it once through its recorded replicated write, after every actor
        exists, and then gives each actor its serial with ``set_version``. Until then every
        insert into a class whose rows carry a version serial is refused (``_insert``).
        ``version_label`` is kept only to name the label in that refusal.

        ``replicated_tables`` are the classes the pool replicates. The prune at startup of a
        replicated class is the pool's, under a record, before any actor exists
        (``ShardedPool._prune_replicated_tables``); for those classes ``_validate_on_startup``
        here reports only, whatever ``prune_unvalidated`` is.
        """
        self._timeout = timeout
        self._my_name = my_name

        self._version_label: str = version_label
        # the version row's serial, set by ShardedPool through set_version once the row exists
        self._version_serial: Optional[int] = None

        self._replicated_tables = frozenset(replicated_tables)

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

        self._read_table_config: Optional[ReadTableConfigType] = read_table_config

    def set_version(self, serial: int):
        """
        Set the serial of the version row, which ShardedPool has found or written on every shard
        (prompts/a3-v2-readiness, prompt 02). Every later insert into a class with a ``version``
        column carries it. Setting it again to the same serial is harmless; setting a different
        one raises, since rows already inserted carry the first.
        """
        if isinstance(serial, bool) or not isinstance(serial, int):
            raise TypeError(
                f"Datastore.set_version: the version serial must be an int, not {serial!r}"
            )
        if self._version_serial is not None and self._version_serial != serial:
            raise RuntimeError(
                f'Datastore "{self._my_name}": the version serial is already {self._version_serial} '
                f'(label "{self._version_label}"), and cannot be changed to {serial}'
            )
        self._version_serial = serial

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
        for cls_name in self._factories:
            if cls_name in self._schema:
                raise RuntimeWarning(
                    f"Duplicate registered factory for storable class '{cls_name}'"
                )

        # the tables and schema records come from the one schema builder (Datastore/SQL/schema.py);
        # the actor adds only the inserters, which are bound to its own _insert
        built = build_schema(self._metadata, self._factories)

        for cls_name, schema in built.records.items():
            tab = schema["table"]
            if tab is not None:
                # build inserter
                inserter = functools.partial(self._insert, schema, tab)
                schema["insert"] = inserter

                # also store table and inserter in their own separate cache
                self._tables[cls_name] = tab
                self._inserters[cls_name] = inserter
            else:
                schema["insert"] = None

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
        if actions is None:
            return

        if not isinstance(actions, Iterable):
            raise RuntimeError("Could not interpret actions type")

        actions = set(actions)
        dropped = set()
        for action in _drop_order:
            if action in actions:
                print(f'Datastore: dropping tables for "{action}"')
                drop_list = _drop_actions[action]
                for table_name in drop_list:
                    if self._inspector.has_table(table_name):
                        tab = self._tables.get(table_name, None)
                        if tab is not None:
                            tab.drop(self._engine)
                            self._metadata.remove(tab)
                dropped.add(action)

        for action in dropped:
            while action in actions:
                actions.remove(action)

        for action in actions:
            print(f'Datastore: unknown drop action "{action}"')

        # regenerate inspector to pick up any changes that were made
        # (the inspector apparently does not automatically reflect dropped tables)
        self._inspector = sqla.inspect(self._engine)

    def _validate_on_startup(self):
        """
        Report the unvalidated rows of every class that validates at startup, and under
        ``prune_unvalidated`` prune those of the sharded classes. A replicated class is never
        pruned here: its prune is the pool's, under a record, made before this actor existed
        (prompts/a3-v2-readiness, prompt 02), and an actor that deleted a replicated row from its
        own shard would leave the shards different with no record to say which is right.
        """
        printed_header = False

        for cls_name, record in self._schema.items():
            if record["validate_on_startup"]:
                factory = self._factories[cls_name]

                tab = record["table"]

                prune = (
                    self._prune_unvalidated and cls_name not in self._replicated_tables
                )

                with self._engine.begin() as conn:
                    msgs = factory.validate_on_startup(conn, tab, self._tables, prune)

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

    def object_get(
        self, ObjectClass, *, insert_timestamp: Optional[datetime] = None, **kwargs
    ):
        """
        Get-or-insert one object (keyword payload) or several (``payload_data``).

        ``insert_timestamp``, when given, is the timestamp every row this call inserts is stamped
        with, in place of ``datetime.now()``. ShardedPool passes one per replicated write, the
        same to every shard, so that copies of a replicated row are identical
        (prompts/datastore-integrity, prompt 01). It is a per-call argument and never actor state.
        Without it the call behaves exactly as before; no sharded write passes it.
        """
        if isinstance(ObjectClass, str):
            cls_name = ObjectClass
        else:
            cls_name = ObjectClass.__name__

        profile_metadata = {"object": cls_name}
        if "payload_data" in kwargs:
            profile_metadata.update({"type": "vector"})
        else:
            profile_metadata.update({"type": "scalar"})

        with ProfileBatchManager(
            self._profile_batcher, "object_get", profile_metadata
        ) as mgr:
            self._ensure_registered_schema(cls_name)
            record = self._schema[cls_name]

            tab = record["table"]

            # the inserters for this call: the actor's own, or bound to the override
            inserters = self._inserters_for(insert_timestamp)
            inserter = inserters.get(cls_name, record["insert"])

            # obtain type of factory class for this storable
            factory = self._factories[cls_name]

            if "payload_data" in kwargs:
                payload_data = kwargs["payload_data"]
                scalar = False
            else:
                payload_data = [kwargs]
                scalar = True

            num_items = len(payload_data)
            if num_items > 1:
                mgr.update_num_items(num_items)

            try:
                with self._engine.begin() as conn:
                    objects = [
                        factory.build(
                            payload=p,
                            conn=conn,
                            table=tab,
                            inserter=inserter,
                            tables=self._tables,
                            inserters=inserters,
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

        # for obj in objects:
        #     obj_pickled = cloudpickle.dumps(obj)
        #     print(
        #         f'## Datastore.object_get: serialized size of object type "{type(obj).__name__}" = {humanize.naturalsize(len(obj_pickled))}'
        #     )

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
                    num_items = len(objects)
                    mgr.update_num_items(num_items)

            except SQLAlchemyError as e:
                print(
                    f"!! Database error in datastore build() [store={self._my_name}, physical store={self._db_file}]"
                )
                print(f"|  payload data = {payload}")
                print(f"|  {e}")
                raise e

        return objects

    def object_store(self, objects, *, insert_timestamp: Optional[datetime] = None):
        """
        Store one object or a list of them.

        ``insert_timestamp`` is as for object_get: the timestamp of every row this call inserts,
        in place of ``datetime.now()``, passed by ShardedPool for a replicated store and by
        nothing else. A replica's store() may raise ReplicationMismatch, which does not know its
        shard; it is re-raised here naming this actor.
        """
        if isinstance(objects, list) or isinstance(objects, tuple):
            payload_data = objects
            scalar = False
        else:
            payload_data = [objects]
            scalar = True

        num_items = len(payload_data)
        with ProfileBatchManager(
            self._profile_batcher,
            "object_store",
            num_items=num_items,
        ) as mgr:
            store_data = {}
            output_objects = []
            inserters = self._inserters_for(insert_timestamp)
            with self._engine.begin() as conn:
                for obj in payload_data:
                    cls_name = type(obj).__name__
                    if cls_name not in store_data:
                        store_data[cls_name] = {"number": 0, "time": 0.0}
                    cls_data = store_data[cls_name]

                    with WallclockTimer() as item_timer:
                        self._ensure_registered_schema(cls_name)
                        record = self._schema[cls_name]

                        tab = record["table"]
                        inserter = inserters.get(cls_name, record["insert"])

                        factory = self._factories[cls_name]

                        try:
                            stored = factory.store(
                                obj,
                                conn=conn,
                                table=tab,
                                inserter=inserter,
                                tables=self._tables,
                                inserters=inserters,
                            )
                        except ReplicationMismatch as e:
                            raise e.with_shard(self._my_name)
                        output_objects.append(stored)

                    cls_data["number"] += 1
                    cls_data["time"] += item_timer.elapsed

                conn.commit()

            for cls_data in store_data.values():
                cls_data["time_per_item"] = cls_data["time"] / cls_data["number"]
            mgr.update_metadata(store_data)

        if scalar:
            return output_objects[0]

        return output_objects

    def _inserters_for(self, insert_timestamp: Optional[datetime]):
        """
        The inserters for one call. With no timestamp override they are the actor's own, exactly
        as before. With one, every table's inserter is bound to it for this call only: the
        override travels in the call's arguments and is never kept on the actor, so one call's
        timestamp cannot leak into the next.
        """
        if insert_timestamp is None:
            return self._inserters

        return {
            cls_name: functools.partial(
                self._insert,
                self._schema[cls_name],
                tab,
                insert_timestamp=insert_timestamp,
            )
            for cls_name, tab in self._tables.items()
        }

    def _insert(
        self,
        schema,
        table,
        conn,
        payload,
        insert_timestamp: Optional[datetime] = None,
    ):
        if table is None:
            raise RuntimeError(f"Attempt to insert into null table (schema='{schema}')")

        uses_serial = schema.get("use_serial", True)
        uses_timestamp = schema.get("use_timestamp", False)
        uses_version = schema.get("use_version", False)
        uses_stepping = schema.get("use_stepping", False)

        cls_name = schema["name"]

        # a row that carries a version serial cannot be written before the pool has set it
        # (set_version). Refused here, before the serial lease below, so that a refused insert
        # takes no serial from the broker and writes nothing
        if uses_version and self._version_serial is None:
            raise RuntimeError(
                f'Datastore "{self._my_name}": cannot insert into "{cls_name}" before the version '
                f"serial is set: ShardedPool calls set_version once the version row of label "
                f'"{self._version_label}" exists on every shard'
            )

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
                payload = payload | {"version": self._version_serial}
            if uses_timestamp:
                # a replicated write stamps every shard with the driver's one timestamp
                # (object_get / object_store); every other insert stamps its own time
                timestamp = (
                    insert_timestamp if insert_timestamp is not None else datetime.now()
                )
                payload = payload | {"timestamp": timestamp}
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

    def read_table(self, cls, *args, **kwargs):
        """
        Provide a generic reusable service to scan a table in the underlying datastore
        :param cls:
        :param args:
        :param kwargs:
        :return:
        """
        if self._read_table_config is None:
            raise RuntimeError("Datastore: the read_table service is not configured")

        if isinstance(cls, str):
            class_name = cls
        else:
            class_name = cls.__name__

        if class_name not in self._read_table_config:
            raise RuntimeError(
                f'Datastore: the read_table service is not available for objects of class "{class_name}"'
            )

        with ProfileBatchManager(
            self._profile_batcher, f"read_table[{class_name}]"
        ) as mgr:
            self._ensure_registered_schema(class_name)
            record = self._schema[class_name]

            tab = record["table"]
            factory = self._factories[class_name]

            if not hasattr(factory, "read_table"):
                raise RuntimeError(
                    f'Datastore: the object factory for "{class_name}" does not provide a read_table service'
                )

            config = self._read_table_config[class_name]
            if config.get("tables_arg", False):
                kwargs["tables"] = self._tables

            with self._engine.begin() as conn:
                objects = factory.read_table(conn, tab, *args, **kwargs)

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
