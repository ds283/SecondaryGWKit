import functools
import sqlite3
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Union, Mapping, Callable, Optional, List, Iterable, Dict, Any

import ray
import sqlalchemy as sqla
from ray.actor import ActorHandle
from sqlalchemy.exc import SQLAlchemyError

from Datastore.SQL.ClientPool import SerialPoolManager, SerialLeaseManager
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from Datastore.SQL.ProfileAgent import ProfileBatcher, ProfileBatchManager
from Datastore.SQL.schema import build_schema, drop_order
from Datastore.replication import ReplicationMismatch
from Datastore.replication import ReadOnlyMiss, ReadOnlyWrite
from utilities import WallclockTimer

VERSION_ID_LENGTH = 64


PathType = Union[str, PathLike]


_FactoryMappingType = Mapping[str, SQLAFactoryBase]
_TableMappingType = Mapping[str, sqla.Table]
_InserterMappingType = Mapping[str, Callable]

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
        drop_tables: Optional[List[str]] = None,
        read_table_config: Optional[ReadTableConfigType] = None,
        read_only: bool = False,
        *,
        factories: _FactoryMappingType,
        serial_batch_sizes: Optional[Mapping[str, int]] = None,
    ):
        """
        Initialize an SQL datastore object.

        ``factories`` is the registry of storable classes, name -> factory, that the client gives
        (prompts/datastore-generic, prompt 06): the actor registers it and builds every table from
        it, and imports none. ``serial_batch_sizes`` maps a table to how many serials the actor
        leases from the broker at a time; a table it does not name leases ``SerialPoolManager``'s
        default. ``drop_tables`` names the tables to drop when an existing file is opened
        read-write, in any order: the actor drops them in ``schema.drop_order``'s order, and refuses
        a name its registry does not declare as a table before dropping anything (``_drop_tables``).

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

        ``read_only`` (prompts/a3-v2-readiness, prompt 03; ShardedPool passes it) opens the file
        ``mode=ro`` and writes nothing: no file is created, no table is dropped or created, and
        ``_validate_on_startup`` only reports (``_open_read_only``). Every factory's ``build`` is
        handed an inserter that raises ``ReadOnlyMiss`` instead of inserting
        (``_refuse_insert``); ``object_store`` and ``object_validate`` raise ``ReadOnlyWrite``;
        and a write that still reaches SQLite, which the ``mode=ro`` file refuses, is re-raised as
        ``ReadOnlyWrite`` (``_read_only_refusal``).
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
            profiler=self._profile_batcher,
            broker=self._serial_broker,
            batch_sizes=serial_batch_sizes,
        )

        self._db_file = Path(db_name).resolve()

        # initialize set of registered storable class adapters
        self._factories: _FactoryMappingType = {}
        self.register_factories(factories)

        # initialize empty dict of storage schema
        # each record collects SQLAlchemy column and table definitions, queries, etc., for a registered storable class factories
        self._tables: _TableMappingType = {}
        self._inserters: _InserterMappingType = {}
        self._schema = {}

        # A READ-ONLY ACTOR (prompts/a3-v2-readiness, prompt 03): nothing below this block runs
        self._read_only: bool = bool(read_only)
        if self._read_only:
            self._open_read_only(drop_tables)
            self._read_table_config = read_table_config
            return

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
            self._drop_tables(drop_tables)
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

    # A READ-ONLY ACTOR (prompts/a3-v2-readiness, prompt 03)
    #
    # Built by ShardedPool(read_only=True) after the pool has refused journals, compared the
    # replicated tables and found the version row, all on mode=ro files. The actor opens its own
    # file mode=ro through store_reader's URL, creates nothing, drops nothing, creates no table,
    # and validates at startup in report-only form. It never receives the version serial
    # (set_version is not called), so even an insert that got past the refusing inserters would
    # be refused by _insert before a serial is leased; and the mode=ro file refuses any write that
    # reaches SQLite, which _read_only_refusal re-raises as ReadOnlyWrite.

    def _open_read_only(self, drop_tables):
        """The read-only constructor. Writes nothing; refuses before opening anything if asked
        to drop or prune, and refuses a file that does not exist rather than create it.
        """
        if drop_tables is not None and len(list(drop_tables)) > 0:
            raise ReadOnlyWrite(
                self._my_name,
                f"drop_tables={sorted(drop_tables)}",
                detail="dropping a table is a write",
            )
        if self._prune_unvalidated:
            raise ReadOnlyWrite(
                self._my_name,
                "prune_unvalidated=True",
                detail="the prune at startup deletes unvalidated rows",
            )
        if self._db_file.is_dir() or not self._db_file.is_file():
            raise RuntimeError(
                f'Datastore "{self._my_name}" was opened read-only, and its database file '
                f'"{str(self._db_file)}" is not an existing file. A read-only datastore creates '
                "nothing. Nothing was written"
            )

        # store_reader imports this module at module scope, so it is imported here
        from Datastore.store_reader import read_only_url

        connect_args = {}
        if self._timeout is not None:
            connect_args["timeout"] = self._timeout
        self._engine = sqla.create_engine(
            read_only_url(self._db_file),
            future=True,
            connect_args=connect_args,
        )
        self._metadata = sqla.MetaData()

        self._build_schema()

        # every inserter, the actor's own and the one each schema record carries, refuses
        self._inserters = {
            cls_name: functools.partial(self._refuse_insert, self._schema[cls_name])
            for cls_name in self._tables
        }
        for cls_name, inserter in self._inserters.items():
            self._schema[cls_name]["insert"] = inserter

        self._validate_on_startup_read_only()

    def _validate_on_startup_read_only(self):
        """
        ``_validate_on_startup`` in report-only form: every class that validates at startup is
        read with ``prune=False``, on a connection that is rolled back.
        """
        printed_header = False

        for cls_name, record in self._schema.items():
            if not record["validate_on_startup"]:
                continue

            factory = self._factories[cls_name]
            tab = record["table"]

            try:
                with self._engine.connect() as conn:
                    try:
                        msgs = factory.validate_on_startup(
                            conn, tab, self._tables, False
                        )
                    finally:
                        conn.rollback()
            except SQLAlchemyError as e:
                self._read_only_refusal(e, "validate_on_startup", cls_name)
                raise

            if len(msgs) == 0:
                continue

            if not printed_header:
                if self._my_name is not None:
                    print(
                        f"!! INTEGRITY WARNING ({datetime.now().replace(microsecond=0).isoformat()}): "
                        f'datastore "{self._my_name}" (physical file {str(self._db_file)}), opened '
                        "read-only: reported, not pruned"
                    )
                    printed_header = True

            for line in msgs:
                print(line)

    def _refuse_insert(
        self,
        schema,
        conn,
        payload,
        insert_timestamp: Optional[datetime] = None,
    ):
        """
        The inserter of a read-only actor, in place of ``_insert``, for every table. It raises
        before any ``INSERT`` is issued and before a serial is leased. A row of a replicated class
        is what a lookup inserts on a miss (``store_tag``, ``LambdaCDM``, ``QCD_Cosmology``,
        ``tolerance``, ``GkSourcePolicy``, and ``redshift``/``wavenumber`` when created), so it is a
        ``ReadOnlyMiss``. Any other row (a sharded value row that a vectorized get builds, a tag
        row) is a write of computed data, so it is a ``ReadOnlyWrite``.
        """
        cls_name = schema["name"]
        if cls_name in self._replicated_tables:
            raise ReadOnlyMiss(cls_name, dict(payload), store=self._my_name)
        raise ReadOnlyWrite(
            self._my_name,
            "an insert",
            class_name=cls_name,
            detail=f"payload {dict(payload)!r}",
        )

    def _read_only_refusal(self, e: SQLAlchemyError, operation: str, cls_name=None):
        """
        The backstop. On a read-only actor, if ``e`` is SQLite's refusal of a write to the
        ``mode=ro`` file (``SQLITE_READONLY``, "attempt to write a readonly database"), raise
        ``ReadOnlyWrite`` from it, naming this actor. Anything else, ``no such table`` and a hot
        journal's ``SQLITE_READONLY_ROLLBACK`` included, is left for the caller to re-raise.
        """
        if not getattr(self, "_read_only", False):
            return
        orig = getattr(e, "orig", None)
        if (
            isinstance(orig, sqlite3.OperationalError)
            and getattr(orig, "sqlite_errorcode", None) == sqlite3.SQLITE_READONLY
        ):
            raise ReadOnlyWrite(
                self._my_name,
                operation,
                class_name=cls_name,
                detail=f"SQLite refused the write to the read-only file "
                f'"{str(self._db_file)}": {orig}',
            ) from e

    def read_only_state(self) -> dict:
        """Whether this actor is read-only. ShardedPool calls it on every actor of a read-only
        pool once, and keeps it because the call waits for every constructor: a constructor that
        raises surfaces there, before the pool is returned. A shard that lacks a table, or differs
        otherwise from the declared tables, was refused by the pool before any actor was built.
        """
        return {
            "read_only": getattr(self, "_read_only", False),
        }

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

    def _drop_tables(self, tables):
        """
        Drop the named tables that this shard holds (prompts/datastore-generic, prompt 06).

        A name the registry does not declare as a table raises ``RuntimeError``, naming it, before
        anything is dropped. The declared tables are dropped in ``drop_order``'s order, derived
        from their foreign keys: a table that references another of them is dropped before it. A
        table the file lacks is skipped (an interrupted earlier drop leaves some absent).
        """
        if tables is None:
            return

        if not isinstance(tables, Iterable) or isinstance(tables, str):
            raise RuntimeError("Could not interpret drop_tables type")

        names = list(dict.fromkeys(tables))
        undeclared = [name for name in names if name not in self._tables]
        if len(undeclared) > 0:
            raise RuntimeError(
                f'Datastore "{self._my_name}": cannot drop {undeclared}, which the registry does '
                f"not declare as tables. Nothing was dropped"
            )

        for tab in drop_order([self._tables[name] for name in names]):
            if self._inspector.has_table(tab.name):
                print(f'Datastore: dropping table "{tab.name}"')
                tab.drop(self._engine)
                self._metadata.remove(tab)

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
                # a read-only actor: SQLite's refusal of a write is a ReadOnlyWrite (prompt 03)
                self._read_only_refusal(e, "object_get", cls_name)
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
                # a read-only actor: SQLite's refusal of a write is a ReadOnlyWrite (prompt 03)
                self._read_only_refusal(e, "object_read_batch", cls_name)
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
        # a read-only actor stores nothing (prompts/a3-v2-readiness, prompt 03)
        if getattr(self, "_read_only", False):
            items = objects if isinstance(objects, (list, tuple)) else [objects]
            raise ReadOnlyWrite(
                self._my_name,
                "object_store",
                class_name=", ".join(sorted({type(o).__name__ for o in items})),
            )
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
        # a read-only actor's inserters all refuse, whatever the timestamp (prompt 03)
        if getattr(self, "_read_only", False):
            return self._inserters
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
        # a read-only actor validates nothing (prompts/a3-v2-readiness, prompt 03)
        if getattr(self, "_read_only", False):
            items = objects if isinstance(objects, (list, tuple)) else [objects]
            raise ReadOnlyWrite(
                self._my_name,
                "object_validate",
                class_name=", ".join(sorted({type(o).__name__ for o in items})),
            )
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
