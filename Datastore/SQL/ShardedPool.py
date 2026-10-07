import errno
import os
import random
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Callable, Mapping, NamedTuple, Tuple

import ray
import sqlalchemy as sqla

from Datastore.SQL import Datastore
from Datastore.SQL.Datastore import PathType, ReadTableConfigType
from Datastore.SQL.ProfileAgent import ProfileAgent
from Datastore.SQL.SerialPoolBroker import SerialPoolBroker
from Datastore.SQL.schema import (
    StoreSchemaMismatch,
    build_schema,
    dependent_tables,
    schema_differences,
)
from Datastore.contract import TAG_TABLE, VERSION_LABEL, VERSION_TABLE
from Datastore.replication import (
    ReplicatedDivergence,
    ReplicationInFlight,
    ReplicationMismatch,
    same_value,
)
from Datastore.replication import ReadOnlyMiss, ReadOnlyWrite
from Datastore.shard_paths import (
    resolve_shard_path,
    shard_file_name,
    shard_file_problem,
)
from config.defaults import DEFAULT_STRING_LENGTH

# the files SQLite keeps beside a database while it is open or was not closed cleanly: the
# rollback journal, and the write-ahead log and its index
_SQLITE_JOURNAL_SUFFIXES = ("-journal", "-wal", "-shm")

# copy_store writes the destination primary under this name first (appended to the destination
# primary's file name) and renames it onto its real name last
_INCOMPLETE_COPY_SUFFIX = ".incomplete-copy"


class _RelocationPlan(NamedTuple):
    """What copy_store / move_store will do, fixed before anything is written."""

    mode: str
    src_primary: Path
    src_files: Dict[int, Path]
    dst_primary: Path
    dst_files: Dict[int, Path]
    dst_records: Dict[int, str]
    temp_primary: Optional[Path]


class _Replication(NamedTuple):
    """
    What follows a controlling shard's answer in a replicated write (ShardedPool._replicated_write):
    the serial to record, how to submit the call to one replica, and how to check its answer.
    """

    # the store_id to write into the in-flight record, or None if there is no single one (a
    # vectorized get)
    store_id: Optional[int]
    # submit(handle, timestamp) -> ref: the call to one replica
    submit: Callable
    # check(shard_id, answer): raise ReplicationMismatch if the replica's answer differs
    check: Callable


class _DeletionPlan(NamedTuple):
    """What closed_store_files / delete_store name, fixed before anything is deleted."""

    primary: Path
    # every shard the primary's `shards` table names, serial -> resolved path, present or not
    recorded: Dict[int, Path]
    # the shards to delete, serial -> path, in ascending serial: all of them, or under resume only
    # those still present
    shards: Dict[int, Path]

    def files(self) -> List[Path]:
        """The files to delete, in the order they are deleted: the shards, then the primary."""
        return [self.shards[s] for s in sorted(self.shards)] + [self.primary]


class ShardedPool:
    """
    ShardedPool manages a pool of datastore actors that cooperate to
    form a sharded SQL database
    """

    def __init__(
        self,
        version_label: str,
        db_name: PathType,
        ShardKeyType,
        ShardKeyStoreIdGetter: Callable,
        replicated_tables: List[str],
        sharded_tables: Dict[str, str],
        timeout: int = None,
        shards: int = 10,
        profile_agent: Optional[ProfileAgent] = None,
        job_name: Optional[str] = None,
        prune_unvalidated: Optional[bool] = False,
        drop_tables: Optional[List[str]] = None,
        read_table_config: Optional[ReadTableConfigType] = None,
        read_only: bool = False,
        *,
        factories: Mapping,
        serial_batch_sizes: Optional[Mapping[str, int]] = None,
    ) -> None:
        """
        Initialize a pool of datastore actors
        :param replicated_tables:
        :param sharded_tables:
        :param ShardKeyStoreIdGetter:
        :param version_label:
        :param read_only: open an existing store read-only, writing nothing to any of its files
            (prompts/a3-v2-readiness, prompt 03; _open_read_only)
        :param drop_tables: the tables every actor drops when it opens an existing store, in
            any order: each actor drops them in the order ``schema.drop_order`` derives from their
            foreign keys. A read-only pool refuses any (``ReadOnlyWrite``); a read-write pool
            refuses a name ``factories`` does not declare as a table, and then a drop that would
            leave another table naming rows of a dropped one: every table
            ``schema.dependent_tables`` derives from ``factories`` (a foreign key or a parent a
            factory's ``inventory_spec`` declares, transitively) must be among them. Both before
            anything is opened
        :param factories: the registry of storable classes, name -> factory, which the client owns
            and gives (prompts/datastore-generic, prompt 06). The pool, its actors and the reader
            build every table from it; the layer imports none. Required, by keyword
        :param serial_batch_sizes: table name -> how many serials an actor leases from the broker
            at a time, given to every actor; a table it does not name leases
            ``SerialPoolManager``'s default
        """
        self._job_name: Optional[str] = job_name
        self._version_label: str = version_label

        self._prune_unvalidated: Optional[List[str]] = prune_unvalidated

        ## THE CLIENT'S REGISTRY

        self._factories: Mapping = dict(factories)
        self._serial_batch_sizes: Optional[Mapping[str, int]] = (
            dict(serial_batch_sizes) if serial_batch_sizes is not None else None
        )
        self._drop_tables: Optional[List[str]] = (
            list(drop_tables) if drop_tables is not None else None
        )

        ## SHARDING CONFIGURATION

        # expected type of shard key
        self._ShardKeyType = ShardKeyType
        self._ShardKeyType_name: str = ShardKeyType.__name__

        # provided getter function to extract store id from a shard key object (or a proxy)
        self._ShardKeyStoreIdGetter = ShardKeyStoreIdGetter

        self._replicated_tables: List[str] = replicated_tables
        self._sharded_tables: Dict[str, str] = sharded_tables

        ## DATABASE CONFIGURATION

        self._db_name: PathType = db_name
        self._timeout: int = timeout
        self._shards: int = max(shards, 1)

        # resolve concerts the supplied db_name to an absolute path, resolving symlinks if necessary
        # this database file will be taken to be the primary database
        self._primary_file: PathType = Path(db_name).resolve()

        # shard_db_files is a map from shard number -> path representing the database on disk.
        # Every path in it is absolute and lies in the primary's directory (Datastore/shard_paths.py)
        self._shard_db_files: Dict[int, PathType] = {}

        # shard_records is a map from shard number -> the filename value stored for it in the
        # primary's `shards` table, as read (existing store) or written (new store). Kept so that
        # an error can quote what the table said, not only where it was taken to point
        self._shard_records: Dict[int, str] = {}

        # shard_keys is a map from key id -> shard id
        self._shard_keys: Dict[int, int] = {}

        self._profile_agent = profile_agent

        # true only while _replicated_write runs; replicated calls never overlap in the driver
        self._replication_active: bool = False

        # what the check at open compared and repaired (_reconcile_replicated_tables); None for a
        # store created by this constructor, which has nothing to compare
        self.reconciliation: Optional[dict] = None

        # what the prune at open of the replicated classes deleted, one entry per table and shard
        # (_prune_replicated_tables); empty unless prune_unvalidated found something to prune
        self.pruned: List[dict] = []

        # the version row of version_label, found or written once every actor exists
        self._version = None

        # A READ-ONLY POOL (prompts/a3-v2-readiness, prompt 03). It opens an existing store with
        # every file mode=ro and writes nothing to any of them; _open_read_only is the whole of its
        # construction, and nothing below this block runs for it
        self._read_only: bool = bool(read_only)
        if self._read_only:
            self._open_read_only(version_label, drop_tables, read_table_config)
            return

        # a drop table the registry does not declare is refused before the primary is created or
        # opened
        self._refuse_undeclared_drop_tables()
        # and so is a drop that would leave another table naming rows of a dropped one
        self._refuse_drop_that_leaves_references()

        # if primary file is absent, all shard databases should be likewise absent
        if self._primary_file.is_dir():
            raise RuntimeError(
                f'Specified database file "{str(self._primary_file)}" is a directory'
            )
        if not self._primary_file.exists():
            # ensure parent directories also exist
            self._primary_file.parents[0].mkdir(exist_ok=True, parents=True)

            for i in range(self._shards):
                # the one naming rule, shared with copy_store/move_store (Datastore/shard_paths.py)
                shard_file = self._primary_file.parent / shard_file_name(
                    self._primary_file, i
                )

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
            # a journal beside the primary is rolled back by the primary's first read-write
            # open, which _read_shard_data makes: note it now, for the reconciliation's record
            primary_journal = os.path.lexists(str(self._primary_file) + "-journal")

            self._create_engine()

            # the primary holds exactly the tables _create_engine declares, or the store is
            # refused by name before anything reads them. This is the primary's first read, so it
            # is what rolls back a hot journal on the primary
            self._refuse_a_primary_that_differs()

            self._read_shard_data()

            # fail closed, before any actor exists: a Datastore actor given a missing shard file
            # creates an empty database there, and the pool would open with nothing in it
            self._check_shard_files()

            # before any actor exists, and with no Ray: roll back hot journals, compare every
            # replicated table across the shards, repair an interrupted replication from its
            # record's controlling shard, and refuse to open on any other difference
            self.reconciliation = self._reconcile_replicated_tables(primary_journal)

            # the prune at open of a replicated class is made here, by the pool, under a record:
            # after the check at open (so that an interrupted write is repaired before anything
            # is judged unvalidated) and before the broker or any actor exists. The actors still
            # prune sharded classes, and only report replicated ones
            if self._prune_unvalidated:
                self.pruned = self._prune_replicated_tables()

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

        # the broker is created only now, so that nothing on the Ray side exists until the shard
        # files have been checked
        self._broker = SerialPoolBroker.options(name="SerialPoolBroker").remote(
            name="SerialPoolBroker"
        )

        # create actor pool of datastores, one for each shard. No actor inserts the version row
        # (it is written below, once every actor exists), so none waits on another. They are
        # built in the order the pool has always built them, the last shard id first, because
        # that order is self._shards' and the controlling shard's draw indexes it
        shard_ids = list(self._shard_db_files.keys())
        shard_ids = shard_ids[-1:] + shard_ids[:-1]
        self._shards = {
            key: Datastore.options(name=f"shard{key:04d}-store").remote(
                version_label=version_label,
                db_name=self._shard_db_files[key],
                replicated_tables=list(self._replicated_tables),
                factories=self._factories,
                serial_batch_sizes=self._serial_batch_sizes,
                timeout=self._timeout,
                my_name=f"shard{key:04d}-store",
                serial_broker=self._broker,
                profile_agent=self._profile_agent,
                prune_unvalidated=self._prune_unvalidated,
                drop_tables=self._drop_tables,
                read_table_config=read_table_config,
            )
            for key in shard_ids
        }

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

        self._read_table_config: Optional[ReadTableConfigType] = read_table_config
        if read_table_config is not None:
            for class_name, config in read_table_config.items():
                if class_name not in self._replicated_tables:
                    raise RuntimeError(
                        f'It is only possible to configure a read-table method for a replicated table (class name="{class_name}")'
                    )

        # THE VERSION ROW, last (prompts/a3-v2-readiness, prompt 02). It is replicated, so it is
        # written as every other replicated row is, through _replicated_write and under its
        # record, and never by an actor's constructor. It is first looked for without writing
        # anything: a replicated get commits and clears a record on the primary even when it
        # finds the row, so a get here would write the primary on every open of a store that
        # already holds the label. Only a label no shard holds is written. Then every actor is
        # given the serial; until then an actor refuses every insert that carries it
        self._version = self._find_version_row(version_label)
        if self._version is None:
            self._version = ray.get(
                self._get_impl_replicated_table(
                    VERSION_TABLE, {VERSION_LABEL: version_label}
                )
            )
        ray.get(
            [
                shard.set_version.remote(self._version.store_id)
                for shard in self._shards.values()
            ]
        )

    def _find_version_row(self, version_label: str):
        """
        The version row of ``version_label``, read from every shard file with a plain ``mode=ro``
        select and nothing written; or None if no shard holds it. For an existing store the check
        at open has compared the version table across the shards already, and a new store's
        shards are empty, so the shards agree; reading every one costs a select each, and makes a
        disagreement raise rather than be resolved by whichever shard was read. A label held twice
        on a shard raises, as the factory's own lookup does.

        The object returned is built by the version table's factory, as an actor's get builds it
        (README §6.2 U4 of prompts/datastore-generic): on a ``mode=ro`` connection to the
        lowest-serial shard that holds the row, with an inserter that raises, which is never
        reached because the row exists, so the factory finds it and inserts nothing.
        """
        found = {}
        for sid, path in sorted(self._shard_db_files.items()):
            conn = sqlite3.connect(f"{Path(path).as_uri()}?mode=ro", uri=True)
            try:
                rows = conn.execute(
                    f'SELECT serial, "{VERSION_LABEL}" FROM "{VERSION_TABLE}" '
                    f'WHERE "{VERSION_LABEL}" = ?',
                    (version_label,),
                ).fetchall()
            finally:
                conn.close()
            if len(rows) > 1:
                raise RuntimeError(
                    f'Sharded datastore "{str(self._primary_file)}": shard {sid} holds the version '
                    f'label "{version_label}" {len(rows)} times: {rows}'
                )
            found[sid] = tuple(rows[0]) if len(rows) == 1 else None

        if len(set(found.values())) > 1:
            raise RuntimeError(
                f'Sharded datastore "{str(self._primary_file)}": the shards disagree on the '
                f'version row of label "{version_label}": '
                + "; ".join(f"shard {sid}: {row}" for sid, row in sorted(found.items()))
            )

        row = next(iter(found.values()), None)
        if row is None:
            return None

        # store_reader imports this module at module scope, so it is imported here
        from Datastore.store_reader import read_only_url

        def refuse_insert(conn, payload):
            raise RuntimeError(
                f'Sharded datastore "{str(self._primary_file)}": building the version row of '
                f'label "{version_label}" would insert {payload!r}, though every shard holds it'
            )

        built = build_schema(sqla.MetaData(), self._factories)
        sid = min(s for s, r in found.items() if r is not None)
        engine = sqla.create_engine(
            read_only_url(Path(self._shard_db_files[sid])),
            future=True,
            poolclass=sqla.pool.NullPool,
        )
        try:
            with engine.connect() as conn:
                try:
                    return self._factories[VERSION_TABLE].build(
                        payload={VERSION_LABEL: version_label},
                        conn=conn,
                        table=built.tables[VERSION_TABLE],
                        inserter=refuse_insert,
                        tables=built.tables,
                        inserters={},
                    )
                finally:
                    conn.rollback()
        finally:
            engine.dispose()

    # THE READ-ONLY POOL (prompts/a3-v2-readiness, prompt 03)
    #
    # ShardedPool(read_only=True) opens an existing store and writes nothing to any of its files.
    # Its construction is _open_read_only, which replaces the whole of the read-write constructor
    # after the attributes are set; the read-write path does not reach it. At open, in order (the
    # audit's M1 table, row by row, is in the prompt's log):
    #
    #   1. drop_tables or prune_unvalidated=True: ReadOnlyWrite, before anything is opened;
    #   2. store_reader.open_read_only, through a function-scope import (store_reader imports this
    #      module): a missing primary is refused and nothing is created, not even its directory; a
    #      journal (-journal, -wal, -shm) beside the primary or any shard is refused by its
    #      presence, an empty one included, before any engine is made; every shard is opened
    #      mode=ro, and a shard whose file differs from the declared tables (a table it lacks
    #      included) is refused with StoreSchemaMismatch, before anything reads its tables. The
    #      reader also reads the primary's replicated_tables record (prompts/datastore-generic,
    #      prompt 06), so a primary whose record cannot be read, or names a class the registry
    #      does not declare, is refused here with the reader's RuntimeError, before step 3's check
    #      of the primary;
    #   3. the pool's own engine is mode=ro; the primary is refused with StoreSchemaMismatch if it
    #      differs from the tables _create_engine declares, and then _read_shard_data and
    #      _check_shard_files run on it as they do read-write;
    #   4. the check at open in read-only form (_check_replicated_tables_read_only): a
    #      replication_in_flight record of any operation, prune included, refuses by name before
    #      anything is compared or completed; the replicated tables are compared on mode=ro
    #      engines and a difference refuses as ReplicatedDivergence. No read-write sqlite3.connect
    #      is made: journals were refused by presence in step 2, so there is none to roll back.
    #      No replication_in_flight table is created and nothing is pruned or repaired;
    #   5. the version row is found by _find_version_row, mode=ro, before any actor exists. A
    #      label no shard holds refuses with ReadOnlyMiss, naming the labels the store holds. It is
    #      never written, and set_version is not called: a read-only actor inserts nothing;
    #   6. no broker, no read_largest_store_ids, no notification: no serial is allocated. Every
    #      actor is built with read_only=True and serial_broker=None, and the pool waits for
    #      every constructor through read_only_state;
    #   7. the read_table_config check, as read-write.
    #
    # Afterwards a replicated object_get goes to one shard's actor, drawn as read_table draws it,
    # with no record and no replication (_get_impl_replicated_table_read_only); object_store,
    # object_validate and a new shard key raise ReadOnlyWrite; the actors' inserters raise
    # ReadOnlyMiss.

    def _open_read_only(self, version_label: str, drop_tables, read_table_config):
        """The read-only constructor (the comment above). Writes nothing to any file."""
        # 1. refused before anything is opened
        if drop_tables is not None and len(list(drop_tables)) > 0:
            raise ReadOnlyWrite(
                self._primary_file,
                f"drop_tables={sorted(drop_tables)}",
                detail="dropping a table is a write",
            )
        if self._prune_unvalidated:
            raise ReadOnlyWrite(
                self._primary_file,
                "prune_unvalidated=True",
                detail="the prune at open deletes unvalidated rows",
            )

        # store_reader imports this module at module scope, so it is imported here
        from Datastore.store_reader import open_read_only, read_only_url

        # 2. store_reader's refusals, through its code: missing primary, journals by presence,
        # the shards table read mode=ro and its files checked, and a shard whose file differs from
        # the declared tables, and the replicated_tables record read. It is entered only for these
        # refusals: nothing it yields is kept
        with open_read_only(self._primary_file, self._factories):
            pass

        # 3. the pool's engine, mode=ro. _create_engine builds the tables this pool reads; the
        # engine it makes is never connected (create_engine opens nothing) and is replaced at once
        self._create_engine()
        self._engine.dispose()
        connect_args = {}
        if self._timeout is not None:
            connect_args["timeout"] = self._timeout
        self._engine = sqla.create_engine(
            read_only_url(self._primary_file), future=True, connect_args=connect_args
        )
        self._refuse_a_primary_that_differs()
        self._read_shard_data()
        self._check_shard_files()

        # 4. the check at open, compare and refuse only
        self.reconciliation = self._check_replicated_tables_read_only(read_only_url)

        num_shards = len(self._shard_db_files)
        if num_shards == 0:
            raise RuntimeError("No shard records were read from the sharded datastore")
        if num_shards != self._shards:
            print(
                f"!! WARNING: number of shards read from database (={num_shards}) does not match specified number of shards (={self._shards})"
            )

        # 5. the version row, found and never written
        self._version = self._find_version_row(version_label)
        if self._version is None:
            present = self._version_labels_present()
            raise ReadOnlyMiss(
                VERSION_TABLE,
                {VERSION_LABEL: version_label},
                store=self._primary_file,
                detail=f"the store holds the version label(s) {present}",
            )

        # 6. no broker, and no serial allocated; every actor read-only
        self._broker = None
        shard_ids = list(self._shard_db_files.keys())
        shard_ids = shard_ids[-1:] + shard_ids[:-1]
        self._shards = {
            key: Datastore.options(name=f"shard{key:04d}-store").remote(
                version_label=version_label,
                db_name=self._shard_db_files[key],
                replicated_tables=list(self._replicated_tables),
                factories=self._factories,
                serial_batch_sizes=self._serial_batch_sizes,
                timeout=self._timeout,
                my_name=f"shard{key:04d}-store",
                serial_broker=None,
                profile_agent=self._profile_agent,
                prune_unvalidated=False,
                drop_tables=None,
                read_table_config=read_table_config,
                read_only=True,
            )
            for key in shard_ids
        }
        states = ray.get(
            [shard.read_only_state.remote() for shard in self._shards.values()]
        )
        self.actor_states: Dict[int, dict] = dict(zip(self._shards.keys(), states))

        # 7. the read_table_config check
        self._read_table_config: Optional[ReadTableConfigType] = read_table_config
        if read_table_config is not None:
            for class_name, config in read_table_config.items():
                if class_name not in self._replicated_tables:
                    raise RuntimeError(
                        f'It is only possible to configure a read-table method for a replicated table (class name="{class_name}")'
                    )

        print(
            f'>> Opened existing sharded datastore "{str(self._primary_file)}" read-only with '
            f'{num_shards} shards (version label "{version_label}", serial '
            f"{self._version.store_id})"
        )

    def _check_replicated_tables_read_only(self, read_only_url: Callable) -> dict:
        """
        The check at open of a read-only pool: compare, and repair nothing. Refuses with
        ReadOnlyWrite if the primary holds a replication_in_flight record of any operation, before
        anything is compared, so that a prune record is refused by name and never completed; and
        with ReplicatedDivergence on any difference between the shards' replicated tables. Every
        connection it makes is mode=ro, through ``read_only_url``. Returns the reconciliation
        record, as _reconcile_replicated_tables does, with nothing repaired or cleared.
        """
        shard_ids = sorted(self._shard_db_files.keys())
        reconciliation = {
            "compared": [],
            "shards": list(shard_ids),
            "hot_journals": [],
            "record": None,
            "controller": None,
            "repaired": [],
            "cleared": False,
            "read_only": True,
        }

        # a write in flight: completing or repairing it is a write, so it refuses, by name
        record = self._read_in_flight_record()
        reconciliation["record"] = record
        if record is not None:
            raise ReadOnlyWrite(
                self._primary_file,
                f'completing the write in flight (operation="{record.get("operation")}", '
                f'class="{record.get("class_name")}", controlling shard='
                f'{record.get("controller_shard")}, store_id={record.get("store_id")}, '
                f'started={record.get("started")})',
                detail="the primary holds this replication_in_flight record, so the shards may "
                "differ; the check at open of a read-write pool completes or repairs it. The "
                "record was not cleared",
            )

        built = build_schema(sqla.MetaData(), self._factories)
        specs = self._replicated_table_specs(built)
        reconciliation["compared"] = [spec["name"] for spec in specs]

        engines = {
            sid: sqla.create_engine(
                read_only_url(Path(self._shard_db_files[sid])),
                future=True,
                poolclass=sqla.pool.NullPool,
            )
            for sid in shard_ids
        }
        try:
            snapshots = {}
            for sid in shard_ids:
                with engines[sid].connect() as conn:
                    try:
                        snapshots[sid] = self._read_replicated_tables(conn, specs)
                    finally:
                        conn.rollback()
        finally:
            for engine in engines.values():
                engine.dispose()

        differences = self._replicated_differences(specs, snapshots)
        if len(differences) > 0:
            raise ReplicatedDivergence(
                self._primary_file,
                self._describe_differences(specs, snapshots, differences),
                None,
            )
        return reconciliation

    def _version_labels_present(self) -> List[str]:
        """Every version label the store holds, in serial order, read mode=ro from each shard
        (the check at open has shown them equal).
        """
        labels = []
        for sid, path in sorted(self._shard_db_files.items()):
            conn = sqlite3.connect(f"{Path(path).as_uri()}?mode=ro", uri=True)
            try:
                rows = conn.execute(
                    f'SELECT "{VERSION_LABEL}" FROM "{VERSION_TABLE}" ORDER BY serial'
                ).fetchall()
            finally:
                conn.close()
            for (label,) in rows:
                if label not in labels:
                    labels.append(label)
        return labels

    def _get_impl_replicated_table_read_only(self, cls_name, kwargs):
        """
        A replicated get on a read-only pool: one shard's actor answers, drawn as read_table
        draws it, with no replication_in_flight record and no replication; _replicated_write is
        never entered. A miss that would insert raises ReadOnlyMiss in the actor. A get of the
        shard-key class keeps _assign_shard_keys, which raises ReadOnlyWrite where it would insert
        a key. Returns the actor's ref, as the read-write get does.
        """
        # we only need to read the row from a single shard, so pick one at random
        shard_ids = list(self._shards.keys())
        i = random.randrange(len(shard_ids))

        # swap this entry with the last element, then pop it
        shard_ids[i], shard_ids[-1] = shard_ids[-1], shard_ids[i]
        shard_key = shard_ids.pop()

        ref = self._shards[shard_key].object_get.remote(cls_name, **kwargs)

        if cls_name == self._ShardKeyType_name:
            self._assign_shard_keys(ray.get(ref))

        return ref

    def _refuse_write(self, operation: str, objects) -> None:
        """Raise ReadOnlyWrite for ``operation`` on a read-only pool, naming the classes."""
        items = objects if isinstance(objects, (list, tuple)) else [objects]
        raise ReadOnlyWrite(
            self._primary_file,
            operation,
            class_name=", ".join(sorted({type(item).__name__ for item in items})),
        )

    @property
    def factories(self) -> Mapping:
        """The registry of storable classes this pool was given (read-only: a copy)."""
        return dict(self._factories)

    def _refuse_undeclared_drop_tables(self) -> None:
        """Refuse a drop table the registry does not declare as a table, naming it, before the
        primary is created or opened. Read-write only: a read-only pool has already refused any
        drop table with ``ReadOnlyWrite`` (``_open_read_only``, step 1)."""
        if not self._drop_tables:
            return
        declared = build_schema(sqla.MetaData(), self._factories).tables
        undeclared = [name for name in self._drop_tables if name not in declared]
        if len(undeclared) > 0:
            raise RuntimeError(
                f'Cannot open sharded datastore "{str(self._primary_file)}": cannot drop '
                f"{undeclared}, which the registry does not declare as tables. Nothing was opened"
            )

    def _refuse_drop_that_leaves_references(self) -> None:
        """Refuse drop tables that another table names, before the primary is created or opened,
        naming the tables given and every table that must be dropped with them. Which tables those
        are is ``schema.dependent_tables``' rule, read from the registry: a foreign key or a parent
        a factory's ``inventory_spec`` declares, followed transitively. Read-write only, and after
        ``_refuse_undeclared_drop_tables``, so every name is a declared table."""
        if not self._drop_tables:
            return
        dependents = [
            name
            for name in dependent_tables(self._drop_tables, self._factories)
            if name not in self._drop_tables
        ]
        if len(dependents) > 0:
            raise RuntimeError(
                f'Cannot open sharded datastore "{str(self._primary_file)}": cannot drop '
                f"{sorted(set(self._drop_tables))} alone: the rows of {dependents} would name "
                f"rows that are gone, so they must be dropped too. Nothing was opened"
            )

    @property
    def primary(self) -> Path:
        """
        The primary database file of this pool, resolved. Read-only: it is what the structured
        inventory (``Datastore.store_inventory.read_inventory``) is given, so that a caller holding
        an open pool does not reach into ``_primary_file``.
        """
        return self._primary_file

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ray.get(
            [
                shard.__exit__.remote(exc_type=None, exc_val=None, exc_tb=None)
                for shard in self._shards.values()
            ]
        )

        if self._profile_agent is not None:
            ray.get(self._profile_agent.clean_up.remote())

        if self._engine is not None:
            self._engine.dispose()

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
        self._shard_key_config_table = sqla.Table(
            "shard_key_config",
            self._metadata,
            sqla.Column(
                "key_type",
                sqla.String(DEFAULT_STRING_LENGTH),
                primary_key=True,
                nullable=False,
            ),
        )
        self._shard_key_table = sqla.Table(
            "shard_keys",
            self._metadata,
            sqla.Column("key_serial", sqla.Integer, primary_key=True, nullable=False),
            sqla.Column(
                "shard_id",
                sqla.Integer,
                sqla.ForeignKey("shards.serial"),
                index=True,
                nullable=False,
            ),
        )
        self._replicated_tables_table = sqla.Table(
            "replicated_tables",
            self._metadata,
            sqla.Column("serial", sqla.Integer, primary_key=True, nullable=False),
            sqla.Column("table", sqla.String(DEFAULT_STRING_LENGTH), nullable=False),
        )
        self._sharded_tables_table = sqla.Table(
            "sharded_tables",
            self._metadata,
            sqla.Column("serial", sqla.Integer, primary_key=True, nullable=False),
            sqla.Column("table", sqla.String(DEFAULT_STRING_LENGTH), nullable=False),
            sqla.Column("key_attr", sqla.String(DEFAULT_STRING_LENGTH), nullable=False),
        )
        # at most one row, naming the replicated write in progress. For a get, store or validate
        # it is committed before the controlling shard's call is submitted and deleted after the
        # last replica's call has returned and been checked (_replicated_write); a row here means
        # the shards may differ, and that the controlling shard's rows of this class are the
        # right ones. For a prune (operation "prune", prompts/a3-v2-readiness prompt 02) it is
        # committed before the first shard is pruned and deleted after the last
        # (_prune_replicated_tables); a prune has no controlling shard, so controller_shard is
        # NULL
        self._replication_in_flight_table = sqla.Table(
            "replication_in_flight",
            self._metadata,
            sqla.Column(
                "operation", sqla.String(DEFAULT_STRING_LENGTH), nullable=False
            ),
            sqla.Column(
                "class_name", sqla.String(DEFAULT_STRING_LENGTH), nullable=False
            ),
            sqla.Column("controller_shard", sqla.Integer, nullable=True),
            sqla.Column("store_id", sqla.Integer, nullable=True),
            sqla.Column("started", sqla.DateTime(), nullable=False),
        )

    def _refuse_a_primary_that_differs(self):
        """
        Refuse, with StoreSchemaMismatch, a primary that does not hold exactly the tables
        _create_engine declares, each with exactly its declared columns. An existing primary is
        written whole by _write_shard_data, so no difference is allowed. Read on the pool's own
        engine (mode=ro for a read-only pool); writes nothing.
        """
        tables = {
            table.name: table
            for table in (
                self._shard_file_table,
                self._shard_key_config_table,
                self._shard_key_table,
                self._replicated_tables_table,
                self._sharded_tables_table,
                self._replication_in_flight_table,
            )
        }
        with self._engine.connect() as conn:
            differences = schema_differences(conn, tables)
            conn.rollback()
        if not differences.empty:
            raise StoreSchemaMismatch(
                "open",
                self._primary_file,
                f'the primary "{str(self._primary_file)}"',
                differences,
            )

    def _write_shard_data(self):
        self._shard_file_table.create(self._engine)
        self._shard_key_config_table.create(self._engine)
        self._shard_key_table.create(self._engine)
        self._replicated_tables_table.create(self._engine)
        self._sharded_tables_table.create(self._engine)
        self._replication_in_flight_table.create(self._engine)

        # each shard is recorded relative to the primary's directory. Every shard is created as a
        # sibling of the primary, so this is its bare file name, and the store stays readable when
        # its directory is moved or copied (Datastore/shard_paths.py)
        shard_file_values = []
        for key, db_name in self._shard_db_files.items():
            db_name = Path(db_name)
            try:
                record = str(db_name.relative_to(self._primary_file.parent))
            except ValueError:
                raise RuntimeError(
                    f'Shard #{key} (database file="{str(db_name)}") is not in the directory of the primary database "{str(self._primary_file)}"'
                )
            # what is written must read back as the same file
            if resolve_shard_path(self._primary_file, record) != db_name:
                raise RuntimeError(
                    f'Shard #{key} (database file="{str(db_name)}") cannot be recorded relative to the primary database "{str(self._primary_file)}" (record="{record}")'
                )
            self._shard_records[key] = record
            shard_file_values.append({"serial": key, "filename": record})

        with self._engine.begin() as conn:
            # write table of database shard files
            conn.execute(sqla.insert(self._shard_file_table), shard_file_values)

            # write shard key configuration type
            conn.execute(
                sqla.insert(self._shard_key_config_table),
                {"key_type": self._ShardKeyType_name},
            )

            # write table of replicated tables
            replicated_table_values = [
                {"serial": n, "table": t} for n, t in enumerate(self._replicated_tables)
            ]
            # SQLAlchemy 2.x executes DEFAULT VALUES when given an empty list;
            # guard to avoid that when no replicated tables are configured.
            if replicated_table_values:
                conn.execute(
                    sqla.insert(self._replicated_tables_table), replicated_table_values
                )

            # write table of sharded tables
            sharded_table_values = [
                {"serial": n, "table": t, "key_attr": k}
                for n, (t, k) in enumerate(self._sharded_tables.items())
            ]
            # SQLAlchemy 2.x executes DEFAULT VALUES when given an empty list;
            # guard to avoid that when no sharded tables are configured.
            if sharded_table_values:
                conn.execute(self._sharded_tables_table.insert(), sharded_table_values)

            conn.commit()

    def _read_shard_data(self):
        with self._engine.begin() as conn:
            # read table of database shard files
            shard_files = conn.execute(
                sqla.select(
                    self._shard_file_table.c.serial,
                    self._shard_file_table.c.filename,
                )
            )
            # shared with copy_store/move_store, which read a source store's records the same way
            ShardedPool._resolve_shard_rows(
                self._primary_file,
                shard_files,
                self._shard_db_files,
                self._shard_records,
            )

            # read shard key configuration type
            shard_key_configs = conn.execute(
                sqla.select(
                    self._shard_key_config_table.c.key_type,
                )
            )
            num_config = 0
            for row in shard_key_configs:
                num_config += 1
                if num_config == 1:
                    if row.key_type != self._ShardKeyType_name:
                        raise RuntimeError(
                            f'Existing ShardedPool was configured with shard key type "{row.key_type}", but provided type was "{self._ShardKeyType_name}"'
                        )

                elif num_config > 1:
                    raise RuntimeError(
                        f'ShardedPool has unexpected multiple shard key types: {num_config}="{row.key_type}"'
                    )
            if num_config == 0:
                raise RuntimeError(f"No configured shard key type was found")
            elif num_config > 1:
                raise RuntimeError(f"Multiple configured shard key types were found")

            # read table of replicated tables
            replicated_table_data = conn.execute(
                sqla.select(
                    self._replicated_tables_table.c.serial,
                    self._replicated_tables_table.c.table,
                )
            )
            missing_read_replicated = set()
            missing_supplied_replicated = set(self._replicated_tables)
            for row in replicated_table_data:
                if row.table not in missing_supplied_replicated:
                    missing_read_replicated.add(row.table)
                missing_supplied_replicated.discard(row.table)
            if len(missing_read_replicated) > 0:
                print(
                    f"The following replicated tables are configured in the existing ShardedPool, but were not supplied to the constructor:"
                )
                for table in missing_read_replicated:
                    print(f"  {table}")
            if len(missing_supplied_replicated) > 0:
                print(
                    f"The following replicated tables were supplied to the constructor, but are not configured in the existing ShardedPool:"
                )
                for table in missing_supplied_replicated:
                    print(f"  {table}")
            if len(missing_read_replicated) > 0 or len(missing_supplied_replicated) > 0:
                raise RuntimeError(
                    f"Mismatch between replicated tables supplied to the constructor and read from the existing ShardedPool"
                )

            # read table of sharded tables
            sharded_table_data = conn.execute(
                sqla.select(
                    self._sharded_tables_table.c.serial,
                    self._sharded_tables_table.c.table,
                    self._sharded_tables_table.c.key_attr,
                )
            )
            missing_read_sharded = set()
            mismatching_key_attr = {}
            missing_supplied_sharded = set(self._sharded_tables.keys())
            for row in sharded_table_data:
                if row.table not in missing_supplied_sharded:
                    missing_read_sharded.add(row.table)
                attr = self._sharded_tables[row.table]
                if row.key_attr != attr:
                    mismatching_key_attr[row.table] = {
                        "supplied": attr,
                        "configured": row.key_attr,
                    }
                missing_supplied_sharded.discard(row.table)
            if len(missing_read_sharded) > 0:
                print(
                    f"The following sharded tables are configured in the existing ShardedPool, but were not supplied to the constructor:"
                )
                for table in missing_read_sharded:
                    print(f"  {table}")
            if len(missing_supplied_sharded) > 0:
                print(
                    f"The following sharded tables are supplied to the constructor, but are not configured in the existing ShardedPool:"
                )
                for table in missing_supplied_sharded:
                    print(f"  {table}")
            if len(mismatching_key_attr) > 0:
                print(
                    f"The following sharded tables were configured with a different key attribute in the existing ShardedPool:"
                )
                for table, data in mismatching_key_attr.items():
                    print(
                        f'  {table}: configured key="{data["configured"]}", supplied key="{data["supplied"]}"'
                    )
            if len(missing_read_sharded) > 0 or len(missing_supplied_sharded) > 0:
                raise RuntimeError(
                    f"Mismatch between sharded tables supplied to the constructor and read from the existing ShardedPool"
                )
            if len(mismatching_key_attr) > 0:
                raise RuntimeError(
                    f"Some sharded tables had mismatching key configurations in the existing ShardedPool"
                )

            # read table of existing shard keys
            keys = conn.execute(
                sqla.select(
                    self._shard_key_table.c.key_serial,
                    self._shard_key_table.c.shard_id,
                )
            )
            for key in keys:
                self._shard_keys[key.key_serial] = key.shard_id

    def _check_shard_files(self):
        """
        Refuse to open an existing pool unless every shard read from the primary is a usable file
        in the primary's directory, and no two shards are the same file.

        Called from the constructor after _read_shard_data() and before any actor is created. It
        exists because a Datastore actor given a missing file creates an empty database there
        (Datastore.py, correct for a single store), so without this check a moved store opens
        with empty shards and every sharded lookup misses, and a mistake in shard path resolution
        would open the wrong store silently instead of raising.
        """
        # shared with copy_store/move_store, which check a source store's shards the same way
        problems = ShardedPool._shard_file_problems(
            self._shard_db_files, self._shard_records
        )

        if len(problems) > 0:
            raise RuntimeError(
                f'Cannot open sharded datastore "{str(self._primary_file)}": '
                + "; ".join(problems)
            )

    # THE REPLICATED TABLES, RECONCILED AT OPEN
    #
    # A replicated write interrupted after its controlling shard committed leaves the shards
    # different and the primary's replication_in_flight record set (_replicated_write). Nothing
    # in the write path deletes a replicated row, with one recorded exception: the prune at open
    # of a replicated class (_prune_replicated_tables), which runs under a record of its own,
    # operation "prune", and is completed below by running it again. The controller commits
    # first and the replicas copy only what it committed, and a flag is only ever turned on, so
    # after an interrupted replication the right state is exactly "every shard holds what the
    # controller holds" (docs/datastore-integrity-audit.md §4.5). The version row is such a
    # replicated row: since prompts/a3-v2-readiness prompt 02 it is written only through
    # _replicated_write, by ShardedPool.__init__, and an interrupted write of it is repaired by
    # the generic get repair below like any other. _reconcile_replicated_tables runs in the
    # constructor after _check_shard_files and before the broker or any actor exists, on plain
    # SQLAlchemy connections to the shard files through tables built by build_schema, with no Ray:
    #
    #   1. each shard is opened read-write once, so that SQLite rolls back a hot journal;
    #   2. a "prune" record is completed and nothing else is done (_complete_interrupted_prune):
    #      the record's class is pruned again on every shard, inside one transaction per shard,
    #      every replicated table is then compared inside those transactions, any difference
    #      refuses with every shard rolled back, and only if they all agree are the shards
    #      committed and the record cleared, last. This is done whatever the open's
    #      prune_unvalidated is: completing an interrupted write is repair, not a choice of this
    #      open. A prune record never reaches the get/store/validate licence below;
    #   3. otherwise every replicated table, and the tag table of every replicated class, is read
    #      from every shard and compared by full row; the primary's shard_keys are checked against
    #      the rows of the shard-key class every shard holds;
    #   4. no record and no difference: nothing is written, and the store opens as before;
    #   5. a get, store or validate record, and differences its interrupted write explains: each
    #      row of the record's class that the record's controlling shard holds and another shard
    #      lacks is copied to that shard verbatim (a class whose unit has more than one table as
    #      a unit: its row, its tag rows and the rows that declare it their owner, _unit_tables),
    #      a monotone flag its factory declares and the controller holds set is set, and after a
    #      validate the class's declared validated column is recomputed on each shard by its
    #      factory's revalidate, the factory's own rule. Every shard is then compared again,
    #      inside the repair's transactions; only if they all agree are the shards committed, and
    #      then the record is cleared, in its own commit, last. A record over identical shards is
    #      simply cleared;
    #   6. anything else raises ReplicatedDivergence, and nothing is written.
    #
    # Apart from completing a recorded prune, which deletes exactly the rows the factory's own
    # rule calls unvalidated, this deletes no row but the record's own and sets no flag to
    # False. Nothing in a sharded table is read, repaired or refused on: those are not
    # replicated, and nothing here can say what they should be.

    def _reconcile_replicated_tables(self, primary_journal: bool = False) -> dict:
        """
        Compare the replicated tables of this existing store across its shards, repair an
        interrupted replication named by the primary's replication_in_flight record, or
        complete an interrupted prune named by it, and raise ReplicatedDivergence on any other
        difference. Return the reconciliation record: the tables compared, the files that had a
        hot journal, the in-flight record found (if any) and its controlling shard (None for a
        prune), what was repaired or pruned, and whether the record was cleared.
        """
        shard_ids = sorted(self._shard_db_files.keys())
        reconciliation = {
            "compared": [],
            "shards": list(shard_ids),
            "hot_journals": [str(self._primary_file)] if primary_journal else [],
            "record": None,
            "controller": None,
            "repaired": [],
            "cleared": False,
        }

        # 1. hot journals: what any read-write open would do, done here so that the comparison
        # reads committed state
        for sid in shard_ids:
            path = Path(self._shard_db_files[sid])
            journal = Path(str(path) + "-journal")
            had_journal = os.path.lexists(journal)
            conn = sqlite3.connect(str(path))
            try:
                conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
            finally:
                conn.close()
            if had_journal:
                reconciliation["hot_journals"].append(str(path))
        for path in reconciliation["hot_journals"]:
            print(
                f'>> Rolled back the journal left beside "{path}" before comparing the replicated tables'
            )

        built = build_schema(sqla.MetaData(), self._factories)

        # 1b. every shard holds the declared tables with exactly their declared columns, or the
        # store is refused by name, before the record is read and before anything is compared or
        # written
        self._refuse_shards_that_differ(built, shard_ids)

        specs = self._replicated_table_specs(built)
        reconciliation["compared"] = [spec["name"] for spec in specs]

        record = self._read_in_flight_record()
        reconciliation["record"] = record
        if record is not None:
            reconciliation["controller"] = record.get("controller_shard")

        # 2. an interrupted prune is completed by running it again, never by the repair below
        if record is not None and record.get("operation") == "prune":
            return self._complete_interrupted_prune(
                record, reconciliation, built, specs, shard_ids
            )

        engines = {
            sid: self._reconcile_engine(self._shard_db_files[sid]) for sid in shard_ids
        }
        conns = {}
        try:
            for sid in shard_ids:
                conns[sid] = engines[sid].connect()

            # 3. the comparison
            snapshots = {
                sid: self._read_replicated_tables(conns[sid], specs)
                for sid in shard_ids
            }
            for conn in conns.values():
                conn.rollback()
            differences = self._replicated_differences(specs, snapshots)

            # 4. no record: any difference refuses
            if record is None:
                if len(differences) > 0:
                    raise ReplicatedDivergence(
                        self._primary_file,
                        self._describe_differences(specs, snapshots, differences),
                        None,
                    )
                return reconciliation

            # 5. a record: what its interrupted write explains is repaired, from its controller
            plan, refusals = self._plan_replicated_repair(
                record, specs, snapshots, differences, conns
            )
            if len(refusals) > 0:
                raise ReplicatedDivergence(self._primary_file, refusals, record)

            actions = []
            if plan["empty"]:
                # nothing to write: what remains is what the plan left to the second comparison
                remaining = [d for d in differences if d["kind"] == "shard_key"]
                if len(remaining) > 0:
                    raise ReplicatedDivergence(
                        self._primary_file,
                        self._describe_differences(specs, snapshots, remaining),
                        record,
                    )
            else:
                for conn in conns.values():
                    conn.begin()
                try:
                    actions, refusals = self._apply_replicated_repair(
                        plan, record, specs, built, conns, snapshots
                    )
                    if len(refusals) > 0:
                        raise ReplicatedDivergence(
                            self._primary_file, refusals, record, after_repair=True
                        )

                    # the second comparison, of what the repair's transactions hold
                    after = {
                        sid: self._read_replicated_tables(conns[sid], specs)
                        for sid in shard_ids
                    }
                    remaining = self._replicated_differences(specs, after)
                    if len(remaining) > 0:
                        raise ReplicatedDivergence(
                            self._primary_file,
                            self._describe_differences(specs, after, remaining),
                            record,
                            after_repair=True,
                        )
                except BaseException:
                    for conn in conns.values():
                        conn.rollback()
                    raise

                for sid in shard_ids:
                    conns[sid].commit()

            # 5, last. every shard holds what the controller holds: the record is cleared
            with self._engine.begin() as conn:
                result = conn.execute(sqla.delete(self._replication_in_flight_table))
                if result.rowcount != 1:
                    raise RuntimeError(
                        f"ShardedPool: the replication_in_flight record has {result.rowcount} "
                        "rows, expected 1"
                    )

            reconciliation["repaired"] = actions
            reconciliation["cleared"] = True
            self._print_reconciliation(record, actions)
            return reconciliation

        finally:
            for conn in conns.values():
                conn.close()
            for engine in engines.values():
                engine.dispose()

    def _refuse_shards_that_differ(self, built, shard_ids) -> None:
        """
        The schema check of a read-write open: compare every shard file with the declared tables
        (``built.tables``), on a mode=ro connection, and refuse with StoreSchemaMismatch the first
        shard, in serial order, that has a table or a column the code does not declare, or that
        lacks a declared column of a table it has. Writes nothing.

        A table a shard lacks is allowed here, and only here (README U1 of
        prompts/datastore-generic): it is what an interrupted first open leaves (each ``create``
        of ``Datastore._ensure_tables`` commits on its own), or an interrupted drop action (an
        actor drops a table and re-creates it). The comparison below reads such a table as empty,
        and the actors complete the open by creating it. That is crash recovery of a store the
        current code wrote, not the reading of an older one.
        """
        # store_reader imports this module at module scope, so it is imported here
        from Datastore.store_reader import read_only_url

        for sid in shard_ids:
            path = Path(self._shard_db_files[sid])
            engine = sqla.create_engine(
                read_only_url(path), future=True, poolclass=sqla.pool.NullPool
            )
            try:
                with engine.connect() as conn:
                    differences = schema_differences(conn, built.tables)
                    conn.rollback()
            finally:
                engine.dispose()
            refused = differences._replace(absent_tables=())
            if not refused.empty:
                raise StoreSchemaMismatch(
                    "open",
                    self._primary_file,
                    f'shard #{sid} "{str(path)}"',
                    differences,
                )

    def _replicated_table_specs(self, built) -> List[dict]:
        """
        What is compared, table by table: every replicated class that has a table, in the order
        the constructor was given them, then the tag table of every replicated class (a table
        that is neither replicated nor sharded, whose foreign keys other than the tag table's all
        name replicated tables).

        Each spec gives the table, the key its rows are compared under and the columns compared.
        A table is keyed by its primary key, and every column is compared, ``timestamp``
        included: every replicated write stamps every shard's copy with its one timestamp. Each
        spec also carries what the table's factory declares (``build_schema``'s record):
        ``owner_column``, ``monotone_flags`` and ``validated_column``.
        """
        replicated = [n for n in self._replicated_tables if n in built.tables]
        listed = set(self._replicated_tables) | set(self._sharded_tables.keys())
        tag_tables = []
        for name, table in built.tables.items():
            if name in listed:
                continue
            parents = {fk.column.table.name for fk in table.foreign_keys} - {TAG_TABLE}
            if len(parents) > 0 and parents <= set(replicated):
                tag_tables.append(name)

        specs = []
        for name in replicated + tag_tables:
            table = built.tables[name]
            record = built.records[name]
            key = tuple(c.name for c in table.primary_key.columns)
            compared = [c.name for c in table.columns]
            specs.append(
                {
                    "name": name,
                    "table": table,
                    "key": key,
                    "compared": compared,
                    "serial_keyed": key == ("serial",),
                    "owner_column": record.get("owner_column"),
                    "monotone_flags": tuple(record.get("monotone_flags", ())),
                    "validated_column": record.get("validated_column"),
                }
            )
        return specs

    def _reconcile_engine(self, path):
        """
        An engine on one shard file for the check at open, with SQLite's own transactions: the
        driver's implicit BEGIN is turned off and one is emitted when a transaction begins, so
        that the repair's transactions and savepoints are real ones.
        """
        connect_args = {}
        if self._timeout is not None:
            connect_args["timeout"] = self._timeout
        engine = sqla.create_engine(
            f"sqlite:///{path}",
            future=True,
            connect_args=connect_args,
            poolclass=sqla.pool.NullPool,
        )

        def on_connect(dbapi_connection, connection_record):
            dbapi_connection.isolation_level = None

        def on_begin(conn):
            conn.exec_driver_sql("BEGIN")

        sqla.event.listen(engine, "connect", on_connect)
        sqla.event.listen(engine, "begin", on_begin)
        return engine

    def _read_in_flight_record(self) -> Optional[dict]:
        """
        The primary's replication_in_flight row, as a dict, or None. More than one row is not
        something the write path leaves, and refuses.
        """
        with self._engine.connect() as conn:
            rows = [
                dict(row)
                for row in conn.execute(
                    sqla.select(self._replication_in_flight_table)
                ).mappings()
            ]
            conn.rollback()
        if len(rows) > 1:
            raise ReplicatedDivergence(
                self._primary_file,
                [
                    {
                        "class_name": "replication_in_flight",
                        "shard": "primary",
                        "key": f"{len(rows)} rows",
                        "detail": f"the table holds at most one row, and holds {rows}",
                    }
                ],
                None,
            )
        return rows[0] if len(rows) == 1 else None

    def _read_replicated_tables(self, conn, specs) -> Dict[str, dict]:
        """
        One shard's replicated rows, table by table: the compared columns, and its rows keyed as
        the spec says, each a tuple in the order of those columns. A table the shard lacks is
        read as empty (an actor would create it empty: README U1 of prompts/datastore-generic).
        The schema check at open (_refuse_shards_that_differ, or the reader for a read-only pool)
        has already refused every other difference, so a table the shard has holds exactly the
        compared columns.
        """
        out = {}
        for spec in specs:
            name = spec["name"]
            present = {
                row[1] for row in conn.exec_driver_sql(f'PRAGMA table_info("{name}")')
            }
            if len(present) == 0:
                out[name] = {"columns": None, "rows": {}, "duplicates": set()}
                continue

            columns = tuple(spec["compared"])
            key_index = [columns.index(k) for k in spec["key"]]
            table = spec["table"]
            rows = {}
            duplicates = set()
            for row in conn.execute(sqla.select(*[table.c[c] for c in columns])):
                values = tuple(row)
                key = tuple(values[i] for i in key_index)
                if key in rows:
                    duplicates.add(key)
                else:
                    rows[key] = values
            out[name] = {"columns": columns, "rows": rows, "duplicates": duplicates}
        return out

    @staticmethod
    def _same_row(a: tuple, b: tuple) -> bool:
        """Two rows' compared values are the same: exactly, with two NaN floats the same value."""
        if a == b:
            return True
        return len(a) == len(b) and all(same_value(x, y) for x, y in zip(a, b))

    def _replicated_differences(self, specs, snapshots) -> List[dict]:
        """
        Every difference between the shards' replicated rows, as dicts by kind:
        ``duplicate`` (a key held by more than one row on a shard), ``missing`` (a key some
        shards hold and others lack), ``differs`` (a key whose compared values differ between
        the shards that hold it) and ``shard_key`` (a key_serial of the primary's shard_keys that
        is not a serial of the shard-key class on every shard).
        """
        shard_ids = sorted(snapshots.keys())
        differences = []
        for spec in specs:
            name = spec["name"]
            tables = {sid: snapshots[sid][name] for sid in shard_ids}

            for sid, t in tables.items():
                for key in sorted(t["duplicates"]):
                    differences.append(
                        {"kind": "duplicate", "table": name, "key": key, "shard": sid}
                    )

            rows = {sid: t["rows"] for sid, t in tables.items()}
            reference = rows[shard_ids[0]] if len(shard_ids) > 0 else {}
            if all(r == reference for r in rows.values()):
                continue

            keys = set()
            for r in rows.values():
                keys.update(r.keys())
            for key in sorted(keys):
                holders = [sid for sid in shard_ids if key in rows[sid]]
                lackers = [sid for sid in shard_ids if key not in rows[sid]]
                if len(lackers) > 0:
                    differences.append(
                        {
                            "kind": "missing",
                            "table": name,
                            "key": key,
                            "holders": holders,
                            "lackers": lackers,
                        }
                    )
                values = {sid: rows[sid][key] for sid in holders}
                first = values[holders[0]]
                if not all(self._same_row(v, first) for v in values.values()):
                    differences.append(
                        {"kind": "differs", "table": name, "key": key, "values": values}
                    )

        # the primary's shard keys name rows of the shard-key class that every shard holds
        key_table = self._ShardKeyType_name
        if len(shard_ids) > 0 and key_table in snapshots[shard_ids[0]]:
            for key_serial, shard_id in sorted(self._shard_keys.items()):
                lackers = [
                    sid
                    for sid in shard_ids
                    if (key_serial,) not in snapshots[sid][key_table]["rows"]
                ]
                if len(lackers) > 0:
                    differences.append(
                        {
                            "kind": "shard_key",
                            "table": "shard_keys",
                            "key": key_serial,
                            "shard_id": shard_id,
                            "lackers": lackers,
                        }
                    )

        return differences

    @staticmethod
    def _key_text(spec: Optional[dict], key) -> str:
        if spec is None or not isinstance(key, tuple):
            return f"key {key}"
        if spec["key"] == ("serial",):
            return f"serial {key[0]}"
        return "key (" + ", ".join(f"{k}={v}" for k, v in zip(spec["key"], key)) + ")"

    def _describe_differences(
        self,
        specs,
        snapshots,
        differences,
        why: Optional[str] = None,
        context: Optional[List[dict]] = None,
    ) -> List[dict]:
        """
        Each difference as the refusal names it: the class (table), the shard or shards that
        deviate, the serial or key, and what differs, followed by ``why`` when it is given. Rows
        missing under different serials whose compared values other than serial and timestamp
        are the same are named as a serial split; ``context``, when given, is the whole list of
        differences those pairs are looked for in.
        """
        by_name = {spec["name"]: spec for spec in specs}

        # a serial split: the same row, but for its serial and timestamp (its two copies come
        # from two writes), held under different serials by different shards
        split_of = {}
        held = {}
        for d in context if context is not None else differences:
            spec = by_name.get(d["table"])
            if d["kind"] != "missing" or spec is None or not spec["serial_keyed"]:
                continue
            snap = snapshots[d["holders"][0]][d["table"]]
            content = tuple(
                v
                for c, v in zip(snap["columns"], snap["rows"][d["key"]])
                if c not in ("serial", "timestamp")
            )
            held.setdefault((d["table"], content), []).append(d)
        for group in held.values():
            if len(group) < 2:
                continue
            for d in group:
                others = [
                    f"serial {o['key'][0]} on shard(s) {o['holders']}"
                    for o in group
                    if o is not d
                ]
                split_of[id(d)] = (
                    f"a serial split: the same row is held under serial {d['key'][0]} on "
                    f"shard(s) {d['holders']} and under " + ", ".join(others)
                )

        out = []
        for d in differences:
            kind = d["kind"]
            spec = by_name.get(d["table"])
            if kind == "duplicate":
                entry = {
                    "class_name": d["table"],
                    "shard": d["shard"],
                    "key": self._key_text(spec, d["key"]),
                    "detail": "more than one row holds this key on the shard",
                }
            elif kind == "missing":
                detail = f"held by shard(s) {d['holders']}, missing on shard(s) {d['lackers']}"
                if id(d) in split_of:
                    detail += f"; {split_of[id(d)]}"
                entry = {
                    "class_name": d["table"],
                    "shard": d["lackers"],
                    "key": self._key_text(spec, d["key"]),
                    "detail": detail,
                }
            elif kind == "differs":
                columns = snapshots[next(iter(d["values"]))][d["table"]]["columns"]
                parts = []
                for i, column in enumerate(columns):
                    groups = []
                    for sid, values in sorted(d["values"].items()):
                        for g in groups:
                            if same_value(g[0], values[i]):
                                g[1].append(sid)
                                break
                        else:
                            groups.append((values[i], [sid]))
                    if len(groups) > 1:
                        parts.append(
                            f"{column}: "
                            + "; ".join(f"{v!r} on shard(s) {s}" for v, s in groups)
                        )
                first = next(iter(d["values"].values()))
                entry = {
                    "class_name": d["table"],
                    "shard": sorted(
                        sid
                        for sid, v in d["values"].items()
                        if not self._same_row(v, first)
                    )
                    or sorted(d["values"].keys()),
                    "key": self._key_text(spec, d["key"]),
                    "detail": "the shards hold different values: " + " | ".join(parts),
                }
            else:  # shard_key
                entry = {
                    "class_name": "shard_keys",
                    "shard": d["lackers"],
                    "key": f"key_serial {d['key']}",
                    "detail": f"the primary assigns {self._ShardKeyType_name} serial "
                    f"{d['key']} to shard {d['shard_id']}, but it is missing on shard(s) "
                    f"{d['lackers']}",
                }
            if why is not None:
                entry["detail"] += f"; {why}"
            out.append(entry)
        return out

    def _plan_replicated_repair(self, record, specs, snapshots, differences, conns):
        """
        Decide, before anything is written, whether the record's interrupted write explains every
        difference, and if it does what to write. Return (plan, refusals); a non-empty refusals
        list names every difference the record does not license repairing.

        The licence: only from the record's controlling shard, only the tables of the record's
        class's unit (``_unit_tables``: its own, its tag tables, and the tables that declare it
        their owner), only rows the controller holds and another shard lacks, written by the
        recorded write (their timestamp is the record's start; under a store_id, that serial), a
        unit of more than one table as a whole, a monotone flag a factory declares that the
        controller holds set after a get, and the class's declared validated column after a
        validate of the record's row, which is recomputed. Anything else is refused.

        Only a get, store or validate record is licensed. A prune record is completed by
        _complete_interrupted_prune and never reaches this; were one passed here, it is refused
        like any other operation. The version row, written since prompts/a3-v2-readiness prompt
        02 only by the pool's recorded get, is admitted by the generic get branch unchanged: its
        table has no timestamp column, so the timestamp test does not apply to it, and only the
        controller's row under the record's store_id (or any, when an interruption before
        _replicated_write's step 3 left store_id null) is copied.
        """
        by_name = {spec["name"]: spec for spec in specs}
        operation = record.get("operation")
        cls_name = record.get("class_name")
        controller = record.get("controller_shard")
        store_id = record.get("store_id")
        started = record.get("started")
        refusals = []

        def refuse(d, why):
            refusals.extend(
                self._describe_differences(
                    specs, snapshots, [d], why=why, context=differences
                )
            )

        if (
            operation not in ("get", "store", "validate")
            or cls_name not in by_name
            or controller not in snapshots
        ):
            return None, [
                {
                    "class_name": cls_name,
                    "shard": controller,
                    "key": "the replication_in_flight record",
                    "detail": "the record does not name an operation, a replicated class and "
                    "a shard of this store, so nothing can say which rows are right",
                }
            ]

        unit = self._unit_tables(cls_name, specs)
        # the flags a get may turn on, and the validated column a validate recomputes, as the
        # factories declare them
        flag_columns = {
            spec["name"]: spec["monotone_flags"]
            for spec in specs
            if len(spec["monotone_flags"]) > 0
        }
        validated_column = by_name[cls_name]["validated_column"]
        flag_names = list(
            dict.fromkeys(f for spec in specs for f in spec["monotone_flags"])
        )
        validated_names = list(
            dict.fromkeys(
                spec["validated_column"]
                for spec in specs
                if spec["validated_column"] is not None
            )
        )
        turned_on_text = (
            f"only a {' or '.join(flag_names)} flag after a get, or "
            f"{' or '.join(validated_names)} after a validate"
        )

        copies = {}  # (table, shard) -> set of keys
        flags = {}  # (table, shard) -> {key: [columns]}
        recompute = False

        for d in differences:
            kind = d["kind"]
            name = d["table"]
            if kind == "shard_key":
                # compared again once the repair is written: a copied row of the shard-key class
                # can satisfy it
                continue
            if kind == "duplicate":
                refuse(d, "this is not something a replication writes")
                continue
            if name not in unit:
                refuse(
                    d,
                    f'the record names "{cls_name}", and an interrupted replication of it '
                    f'writes nothing to "{name}"',
                )
                continue

            if kind == "missing":
                if controller not in d["holders"]:
                    refuse(
                        d,
                        f"the record's controlling shard {controller} lacks this row, and a "
                        "shard can only lack what the controller committed",
                    )
                elif operation == "validate":
                    refuse(d, "a validate writes no row")
                else:
                    for sid in d["lackers"]:
                        copies.setdefault((name, sid), set()).add(d["key"])
                continue

            # differs: each shard against the controller
            if controller not in d["values"]:
                refuse(d, f"the record's controlling shard {controller} lacks this row")
                continue
            columns = snapshots[controller][name]["columns"]
            reference = d["values"][controller]
            for sid, values in sorted(d["values"].items()):
                if sid == controller or self._same_row(values, reference):
                    continue
                differing = [
                    c
                    for c, a, b in zip(columns, values, reference)
                    if not same_value(a, b)
                ]
                turned_on = all(
                    reference[columns.index(c)] is True
                    and values[columns.index(c)] is False
                    for c in differing
                )
                if (
                    name in flag_columns
                    and set(differing) <= set(flag_columns[name])
                    and turned_on
                    and operation == "get"
                    and (store_id is None or d["key"] == (store_id,))
                ):
                    flags.setdefault((name, sid), {})[d["key"]] = differing
                elif (
                    name == cls_name
                    and validated_column is not None
                    and differing == [validated_column]
                    and turned_on
                    and operation == "validate"
                    and d["key"] == (store_id,)
                ):
                    recompute = True
                else:
                    refuse(
                        {**d, "values": {controller: reference, sid: values}},
                        f"{turned_on_text}, can be turned on from the controlling shard "
                        f"{controller}; shard {sid} differs in {differing}",
                    )

        # the controller's full rows, verbatim, for every key to be copied
        rows = {}
        wanted = {}
        for (name, sid), keys in copies.items():
            wanted.setdefault(name, set()).update(keys)
        for name, keys in wanted.items():
            rows[name] = self._read_full_rows(conns[controller], by_name[name], keys)

        # A copy is never checked for the same row held under another serial on the shard it
        # goes to: a shard can hold a row the controller lacks only under a serial the
        # controller does not hold, which the loop above has already refused
        for (name, sid), keys in sorted(copies.items()):
            for key in sorted(keys):
                row = rows[name][key]
                d = {
                    "kind": "missing",
                    "table": name,
                    "key": key,
                    "holders": [controller],
                    "lackers": [sid],
                }
                # written by the recorded write: its timestamp is the record's start, and under
                # a store_id the class's own row is that serial
                if "timestamp" in row and row["timestamp"] != started:
                    refuse(
                        d,
                        f"its timestamp {row['timestamp']} is not the recorded write's start "
                        f"{started}, so the recorded write did not insert it",
                    )
                elif name == cls_name and store_id is not None and key != (store_id,):
                    refuse(d, f"the record names serial {store_id}, not this one")

        # a unit of more than one table is copied as a unit: a shard lacking the class's row
        # receives every one of its unit's rows that name it, and a shard holding any part of it
        # receives nothing. The two refusals keep the words "model row" and "part of this model",
        # which name no table, so that their text is what it has always been
        if len(unit) > 1:
            for sid in sorted(snapshots.keys()):
                owners = {k[0] for k in copies.get((cls_name, sid), set())}
                for name in unit:
                    if name == cls_name:
                        continue
                    # a member row's owner is in the column naming the class's row
                    snap = snapshots[controller][name]
                    owner_index = snap["columns"].index(
                        self._column_naming(by_name[name], cls_name)
                    )
                    owner_of = {
                        k: row[owner_index]
                        for k, row in snap["rows"].items()
                        if row[owner_index] in owners
                    }
                    got = copies.get((name, sid), set())
                    held = set(owner_of.keys())
                    for key in sorted(got - held):
                        refuse(
                            {
                                "kind": "missing",
                                "table": name,
                                "key": key,
                                "holders": [controller],
                                "lackers": [sid],
                            },
                            f"shard {sid} holds its model row, so this is not part of a "
                            f"{cls_name} it lacks",
                        )
                    for key in sorted(held - got):
                        refuse(
                            {
                                "kind": "missing",
                                "table": cls_name,
                                "key": (owner_of[key],),
                                "holders": [controller],
                                "lackers": [sid],
                            },
                            f"shard {sid} already holds part of this model ({name} {key}), "
                            f"and a {cls_name} is copied only as a unit",
                        )

        plan = {
            "copies": copies,
            "flags": flags,
            "recompute": recompute,
            "rows": rows,
            "empty": len(copies) == 0 and len(flags) == 0 and not recompute,
        }
        return plan, refusals

    @staticmethod
    def _read_full_rows(conn, spec, keys) -> Dict[tuple, dict]:
        """The controller's rows under ``keys``, every column of the table, as mappings."""
        table = spec["table"]
        first = spec["key"][0]
        out = {}
        for row in conn.execute(
            sqla.select(*table.columns).filter(
                table.c[first].in_(sorted({k[0] for k in keys}))
            )
        ).mappings():
            key = tuple(row[k] for k in spec["key"])
            if key in keys:
                out[key] = dict(row)
        conn.rollback()
        return out

    def _apply_replicated_repair(self, plan, record, specs, built, conns, snapshots):
        """
        Write the plan, inside each shard's open transaction: the copies (in the specs' order,
        so that a class's row precedes its tag and owned rows), then the flags, then the
        recomputed validated flags. Return (actions, refusals).
        """
        by_name = {spec["name"]: spec for spec in specs}
        controller = record["controller_shard"]
        actions = []
        refusals = []

        for spec in specs:
            name = spec["name"]
            for sid in sorted(conns.keys()):
                keys = plan["copies"].get((name, sid))
                if not keys:
                    continue
                conns[sid].execute(
                    sqla.insert(spec["table"]),
                    [plan["rows"][name][key] for key in sorted(keys)],
                )
                actions.append(
                    {
                        "action": "copied",
                        "class_name": name,
                        "shard": sid,
                        "from": controller,
                        "keys": sorted(keys),
                    }
                )

        for (name, sid), per_key in sorted(plan["flags"].items()):
            table = by_name[name]["table"]
            for key, columns in sorted(per_key.items()):
                conns[sid].execute(
                    sqla.update(table)
                    .where(table.c.serial == key[0])
                    .values({c: True for c in columns})
                )
            actions.append(
                {
                    "action": "flags set",
                    "class_name": name,
                    "shard": sid,
                    "from": controller,
                    "keys": sorted(per_key.keys()),
                    "columns": sorted({c for cols in per_key.values() for c in cols}),
                }
            )

        if plan["recompute"]:
            recomputed, refused = self._recompute_validated(
                record, built, conns, snapshots
            )
            actions.extend(recomputed)
            refusals.extend(refused)

        return actions, refusals

    def _recompute_validated(self, record, built, conns, snapshots):
        """
        After a validate of the record's row, recompute the validated column its class's factory
        declares on each shard by the factory's ``revalidate`` (the rule its ``validate`` applies),
        from that shard's own stored rows, each inside a savepoint. A result that turns the flag
        on is kept; any other is rolled back to the savepoint, so that no flag is written False. A
        shard holding True whose rule gives False refuses. Return (actions, refusals); whether the
        shards then agree is for the second comparison.
        """
        cls_name = record["class_name"]
        factory = self._factories[cls_name]
        table = built.tables[cls_name]
        column = built.records[cls_name]["validated_column"]
        serial = record["store_id"]

        actions = []
        refusals = []
        for sid in sorted(conns.keys()):
            snap = snapshots[sid][cls_name]
            row = snap["rows"].get((serial,))
            if row is None:
                continue
            columns = snap["columns"]
            stored = row[columns.index(column)] is True

            savepoint = conns[sid].begin_nested()
            result = factory.revalidate(serial, conns[sid], table, built.tables)
            if result is True and not stored:
                savepoint.commit()
                actions.append(
                    {
                        "action": "validated recomputed",
                        "class_name": cls_name,
                        "shard": sid,
                        "from": sid,
                        "keys": [(serial,)],
                        "validated": True,
                    }
                )
            else:
                savepoint.rollback()
                if stored and result is not True:
                    refusals.append(
                        {
                            "class_name": cls_name,
                            "shard": sid,
                            "key": f"serial {serial}",
                            "detail": f"the shard holds {column}=True, and the factory's rule "
                            "gives False from its own value rows",
                        }
                    )
        return actions, refusals

    def _print_reconciliation(self, record, actions):
        """One line for the record, one per class and shard for what was written, one for the
        clearing, so that a resumed build's log says what was repaired."""

        def keys_text(keys):
            texts = [str(k[0]) if len(k) == 1 else str(k) for k in keys]
            if len(texts) > 20:
                texts = texts[:10] + [f"... ({len(texts) - 20} more) ..."] + texts[-10:]
            return ", ".join(texts)

        print(
            f'>> Sharded datastore "{str(self._primary_file)}" was opened with a replicated '
            f'write in flight (operation="{record.get("operation")}", '
            f'class="{record.get("class_name")}", controlling shard='
            f'{record.get("controller_shard")}, store_id={record.get("store_id")}, '
            f'started={record.get("started")})'
        )
        if len(actions) == 0:
            print(
                ">>   every shard already holds what the controlling shard holds: nothing was "
                "repaired"
            )
        for a in actions:
            n = len(a["keys"])
            if a["action"] == "copied":
                print(
                    f'>>   copied {n} row(s) of "{a["class_name"]}" from shard {a["from"]} to '
                    f'shard {a["shard"]}: {keys_text(a["keys"])}'
                )
            elif a["action"] == "flags set":
                print(
                    f'>>   set {", ".join(a["columns"])} on {n} row(s) of "{a["class_name"]}" on '
                    f'shard {a["shard"]}, as shard {a["from"]} holds them: {keys_text(a["keys"])}'
                )
            else:
                print(
                    f'>>   recomputed validated of "{a["class_name"]}" serial '
                    f'{a["keys"][0][0]} on shard {a["shard"]} from its own value rows: '
                    f'{a["validated"]}'
                )
        print(">>   the replication_in_flight record was cleared")

    # THE PRUNE AT OPEN OF A REPLICATED CLASS
    #
    # Under prune_unvalidated, the unvalidated rows of every replicated class whose factory
    # validates at startup (with the rows of its unit, _unit_tables) are deleted by the pool, not
    # by the actors (prompts/a3-v2-readiness prompt 02, the user's decision U1). An
    # actor that pruned its own shard in its constructor did so outside any record, so an open
    # interrupted between actors left shards that differ with nothing to say which is right
    # (docs/a3-v2-readiness-audit.md V1, case 4). _prune_replicated_tables runs in the
    # constructor after the check at open and after replication_in_flight exists, before the
    # broker or any actor exists, on plain connections to the shard files, with no Ray. For each
    # such class:
    #
    #   1. every shard is read, writing nothing, by the factory's own validate_on_startup with
    #      prune=False. If no shard reports an unvalidated row, nothing is written and no record
    #      is made, so a pruning open of a store with nothing to prune writes nothing;
    #   2. a record, operation "prune", naming the class, with no controlling shard, is committed
    #      on the primary in its own transaction, before the first shard is touched;
    #   3. each shard, in ascending id, is pruned in its own transaction by the factory's
    #      validate_on_startup with prune=True, so that the rule for "unvalidated" stays in the
    #      factory. Before the transaction commits, the factory is asked again with prune=False
    #      and must report nothing, and every compared table must have lost rows only, and only
    #      the class's own and those of the other tables of its unit. Otherwise the shard is
    #      rolled back and the open raises with the record left set. The factory catches a
    #      SQLAlchemyError from its own deletes and returns normally, so its return says nothing
    #      about whether the prune took;
    #   4. the record is deleted, in its own commit, last.
    #
    # The prune is idempotent: it deletes the class's unvalidated rows, and only those, so a
    # shard already pruned loses nothing more. That is what lets the check at open complete an
    # interrupted one by running it again (_complete_interrupted_prune). No other replicated
    # write can come between: a record blocks every one, and the check at open runs first.

    @staticmethod
    def _replicated_prune_classes(replicated_tables, built) -> List[str]:
        """The replicated classes whose factory validates, and so prunes, at startup, in the
        order the pool was given them: read from the factories' registrations, not listed.
        """
        return [
            name
            for name in replicated_tables
            if name in built.tables
            and built.records[name].get("validate_on_startup", False)
        ]

    @staticmethod
    def _is_tag_table_of(spec: dict, cls_name: str) -> bool:
        """Whether the compared table of ``spec`` is a tag table of ``cls_name``: it has a foreign
        key to the tag table, and every other foreign key it has references ``cls_name``.
        """
        targets = [fk.column.table.name for fk in spec["table"].foreign_keys]
        others = [t for t in targets if t != TAG_TABLE]
        return TAG_TABLE in targets and len(others) > 0 and set(others) == {cls_name}

    @staticmethod
    def _declares_owner(spec: dict, cls_name: str) -> bool:
        """Whether the compared table of ``spec`` declares an ``owner_column`` whose foreign key
        references ``cls_name``."""
        owner = spec["owner_column"]
        if owner is None:
            return False
        return any(
            fk.column.table.name == cls_name
            for fk in spec["table"].c[owner].foreign_keys
        )

    @staticmethod
    def _unit_tables(cls_name: str, specs) -> List[str]:
        """
        The compared tables of ``cls_name``'s unit, read from foreign keys: the class's own
        table, then the compared tables that are its tag tables, then the compared tables whose
        declared ``owner_column`` references it, each group in the specs' order. It is the unit
        the check at open copies as a whole (``_plan_replicated_repair``) and the tables a prune of
        the class may delete from (``_prune_shard``). A table with a foreign key to the class that
        is not its tag table and does not declare it its owner is not in it (prompts/datastore-
        generic README §6.2, the user's choice for prompt 07).
        """
        unit = [cls_name]
        for spec in specs:
            if spec["name"] != cls_name and ShardedPool._is_tag_table_of(
                spec, cls_name
            ):
                unit.append(spec["name"])
        for spec in specs:
            if spec["name"] != cls_name and ShardedPool._declares_owner(spec, cls_name):
                unit.append(spec["name"])
        return unit

    @staticmethod
    def _column_naming(spec: dict, cls_name: str) -> str:
        """The column of a member of ``cls_name``'s unit that names the class's row: its declared
        ``owner_column``, or a tag table's foreign key to the class."""
        if ShardedPool._declares_owner(spec, cls_name):
            return spec["owner_column"]
        return next(
            fk.parent.name
            for fk in spec["table"].foreign_keys
            if fk.column.table.name == cls_name
        )

    def _prune_needed(self, cls_name: str, built, shard_ids) -> bool:
        """Whether any shard holds a row of ``cls_name`` that its factory calls unvalidated,
        read by the factory's own validate_on_startup with prune=False. Writes nothing.
        """
        factory = self._factories[cls_name]
        table = built.tables[cls_name]
        for sid in shard_ids:
            engine = self._reconcile_engine(self._shard_db_files[sid])
            try:
                with engine.connect() as conn:
                    try:
                        if not sqla.inspect(conn).has_table(cls_name):
                            continue
                        if len(
                            factory.validate_on_startup(
                                conn, table, built.tables, prune=False
                            )
                        ):
                            return True
                    finally:
                        conn.rollback()
            finally:
                engine.dispose()
        return False

    def _prune_shard(self, cls_name: str, sid: int, conn, built, specs):
        """
        Prune ``cls_name`` on shard ``sid``, inside ``conn``'s open transaction, by the factory's
        own validate_on_startup with prune=True, and check that it took (the comment above).
        Return (actions, the factory's messages). Raise RuntimeError if it did not take; the
        caller rolls the transaction back. Commits nothing.
        """
        if not sqla.inspect(conn).has_table(cls_name):
            # a shard without the table holds nothing to prune; an actor creates it empty
            return [], []

        factory = self._factories[cls_name]
        table = built.tables[cls_name]
        unit = self._unit_tables(cls_name, specs)
        # the refusal below names the unit as it always has: the class, then the rest of its unit
        # in the specs' order, which is not _unit_tables' order
        unit_named = [cls_name] + [
            spec["name"]
            for spec in specs
            if spec["name"] != cls_name and spec["name"] in unit
        ]
        path = self._shard_db_files[sid]

        before = self._read_replicated_tables(conn, specs)
        msgs = factory.validate_on_startup(conn, table, built.tables, prune=True)
        left = factory.validate_on_startup(conn, table, built.tables, prune=False)
        after = self._read_replicated_tables(conn, specs)

        problems = []
        if len(left) > 0:
            problems.append(
                "the factory still reports unvalidated rows after its prune: "
                + " | ".join(line.strip() for line in left)
            )
        actions = []
        for spec in specs:
            name = spec["name"]
            was, now = before[name]["rows"], after[name]["rows"]
            deleted = sorted(set(was) - set(now))
            added = sorted(set(now) - set(was))
            changed = sorted(
                k for k in set(was) & set(now) if not self._same_row(was[k], now[k])
            )
            if len(added) > 0 or len(changed) > 0:
                problems.append(
                    f'"{name}" gained or changed rows {added + changed}, and a prune only deletes'
                )
            if len(deleted) > 0 and name not in unit:
                problems.append(
                    f'"{name}" lost rows {deleted}, and a prune of "{cls_name}" deletes only '
                    f"from {unit_named}"
                )
            if len(deleted) > 0:
                actions.append(
                    {
                        "action": "pruned",
                        "class_name": name,
                        "shard": sid,
                        "keys": deleted,
                    }
                )

        if len(problems) > 0:
            raise RuntimeError(
                f'Sharded datastore "{str(self._primary_file)}": the prune of "{cls_name}" on '
                f'shard {sid} ("{str(path)}") did not take: '
                + "; ".join(problems)
                + ". "
                "That shard's transaction was rolled back, and the replication_in_flight record "
                "of the prune is left set, so the next open of this store completes the prune "
                "on every shard"
            )
        return actions, msgs

    def _prune_shard_and_commit(self, cls_name: str, sid: int, built, specs):
        """One shard's prune at open, in its own transaction, committed if it took. Return the
        actions."""
        engine = self._reconcile_engine(self._shard_db_files[sid])
        try:
            with engine.connect() as conn:
                conn.begin()
                try:
                    actions, msgs = self._prune_shard(cls_name, sid, conn, built, specs)
                except BaseException:
                    conn.rollback()
                    raise
                conn.commit()
        finally:
            engine.dispose()

        if len(msgs) > 0:
            print(
                f"!! INTEGRITY WARNING ({datetime.now().replace(microsecond=0).isoformat()}): "
                f'sharded datastore "{str(self._primary_file)}", shard {sid} (physical file '
                f'{str(self._shard_db_files[sid])}), pruned by the pool under a "prune" record'
            )
            for line in msgs:
                print(line)
        return actions

    def _commit_prune_record(self, cls_name: str) -> None:
        """Commit the "prune" record of ``cls_name`` on the primary, in its own transaction,
        before any shard is touched. A record already there refuses."""
        record_table = self._replication_in_flight_table
        with self._engine.begin() as conn:
            existing = conn.execute(sqla.select(record_table)).mappings().first()
            if existing is not None:
                raise ReplicationInFlight(self._primary_file, dict(existing))
            conn.execute(
                sqla.insert(record_table),
                {
                    "operation": "prune",
                    "class_name": cls_name,
                    "controller_shard": None,
                    "store_id": None,
                    "started": datetime.now(),
                },
            )

    def _clear_in_flight_record(self, operation: str, cls_name: str) -> None:
        """Delete the primary's one replication_in_flight record, in its own commit."""
        with self._engine.begin() as conn:
            result = conn.execute(sqla.delete(self._replication_in_flight_table))
            if result.rowcount != 1:
                raise RuntimeError(
                    f"ShardedPool: the replication_in_flight record of a {operation} of "
                    f'"{cls_name}" has {result.rowcount} rows, expected 1'
                )

    def _prune_replicated_tables(self) -> List[dict]:
        """
        The prune at open of every replicated class whose factory prunes at startup, each under
        its own "prune" record (the comment above). Return what was deleted, one entry per table
        and shard. Called only under prune_unvalidated.
        """
        built = build_schema(sqla.MetaData(), self._factories)
        specs = self._replicated_table_specs(built)
        shard_ids = sorted(self._shard_db_files.keys())

        pruned = []
        for cls_name in self._replicated_prune_classes(self._replicated_tables, built):
            # 1. nothing to prune on any shard: no record, nothing written
            if not self._prune_needed(cls_name, built, shard_ids):
                continue

            # 2. the record, before the first shard is touched
            self._commit_prune_record(cls_name)

            # 3. each shard in its own transaction
            actions = []
            for sid in shard_ids:
                actions.extend(
                    self._prune_shard_and_commit(cls_name, sid, built, specs)
                )

            # 4. the record, cleared last
            self._clear_in_flight_record("prune", cls_name)
            print(
                f'>> Pruned the unvalidated rows of "{cls_name}" from every shard of '
                f'"{str(self._primary_file)}" under a "prune" record, which was cleared'
            )
            pruned.extend(actions)
        return pruned

    def _complete_interrupted_prune(
        self, record, reconciliation, built, specs, shard_ids
    ) -> dict:
        """
        The check at open's completion of an interrupted prune: the record's class is pruned
        again on every shard, each inside its own open transaction; every replicated table is
        then compared inside those transactions; any difference, or a shard whose prune did not
        take, rolls every shard back and raises with the record left set; otherwise every shard
        is committed and the record cleared, last. Whatever the open's prune_unvalidated is.
        """
        cls_name = record.get("class_name")
        if cls_name not in self._replicated_prune_classes(
            self._replicated_tables, built
        ):
            raise ReplicatedDivergence(
                self._primary_file,
                [
                    {
                        "class_name": cls_name,
                        "shard": None,
                        "key": "the replication_in_flight record",
                        "detail": "the record names a prune of a class that is not a replicated "
                        "class whose factory prunes at startup, so nothing can say what the "
                        "prune should delete",
                    }
                ],
                record,
            )

        engines = {
            sid: self._reconcile_engine(self._shard_db_files[sid]) for sid in shard_ids
        }
        conns = {}
        try:
            for sid in shard_ids:
                conns[sid] = engines[sid].connect()
            for conn in conns.values():
                conn.begin()
            try:
                # 1. the prune, again, on every shard
                actions = []
                for sid in shard_ids:
                    done, _ = self._prune_shard(cls_name, sid, conns[sid], built, specs)
                    actions.extend(done)

                # 2-3. the comparison, of what the prune's transactions hold; any difference
                # refuses
                after = {
                    sid: self._read_replicated_tables(conns[sid], specs)
                    for sid in shard_ids
                }
                remaining = self._replicated_differences(specs, after)
                if len(remaining) > 0:
                    raise ReplicatedDivergence(
                        self._primary_file,
                        self._describe_differences(specs, after, remaining),
                        record,
                        after_repair=True,
                    )
            except BaseException:
                for conn in conns.values():
                    conn.rollback()
                raise

            for sid in shard_ids:
                conns[sid].commit()
        finally:
            for conn in conns.values():
                conn.close()
            for engine in engines.values():
                engine.dispose()

        # 4. the record, cleared last
        self._clear_in_flight_record("prune", cls_name)

        reconciliation["repaired"] = actions
        reconciliation["cleared"] = True

        print(
            f'>> Sharded datastore "{str(self._primary_file)}" was opened with a prune in flight '
            f'(class="{cls_name}", started={record.get("started")}); it was run again on every '
            "shard"
        )
        for a in actions:
            print(
                f'>>   pruned {len(a["keys"])} row(s) of "{a["class_name"]}" on shard '
                f'{a["shard"]}'
            )
        if len(actions) == 0:
            print(">>   every shard was already pruned: nothing more was deleted")
        print(">>   the replication_in_flight record was cleared")
        return reconciliation

    # SHARD RECORDS: READ AND CHECK
    #
    # These two static methods are the one implementation of reading a primary's `shards` rows and
    # checking the files they name. The constructor reaches them through _read_shard_data and
    # _check_shard_files; copy_store and move_store call them directly on a closed store, which has
    # no instance and none of the constructor's arguments.

    @staticmethod
    def _resolve_shard_rows(
        primary_file: Path,
        rows,
        shard_db_files: Dict[int, PathType],
        shard_records: Dict[int, str],
    ) -> None:
        """
        Resolve the (serial, filename) rows of the `shards` table of ``primary_file`` into
        ``shard_db_files`` (serial -> absolute path) and ``shard_records`` (serial -> the stored
        value), which the caller supplies and which are filled in place.

        Every record goes through the one resolver (Datastore/shard_paths.py), which accepts a
        bare file name only. A record that is anything else, an absolute path included, is refused
        as unusable. The rows are not rewritten.
        """
        for serial, stored in rows:
            try:
                filename = resolve_shard_path(primary_file, stored)
            except ValueError as e:
                raise RuntimeError(
                    f'Shard #{serial} of primary database "{str(primary_file)}" has an unusable record: {e}'
                )

            if serial in shard_db_files:
                raise RuntimeError(
                    f'Shard #{serial} already exists (database file="{str(filename)}", existing file="{str(shard_db_files[serial])}")'
                )

            shard_db_files[serial] = filename
            shard_records[serial] = stored

    @staticmethod
    def _shard_file_problems(
        shard_db_files: Dict[int, PathType],
        shard_records: Dict[int, str],
        *,
        missing_ok: bool = False,
    ) -> List[str]:
        """
        Return one message for each resolved shard that is not a usable file (missing, not a
        regular file, a symbolic link: Datastore/shard_paths.py shard_file_problem), and for each
        pair of serials that resolve to the same file. An empty list means every shard is usable.
        """
        problems = []
        for serial, path in sorted(shard_db_files.items()):
            # missing_ok (delete_store(resume=True) only) passes over a shard of which no entry of
            # any kind exists. A dangling symbolic link is an entry, and is still refused, and so is
            # a pair of serials resolving to one file, missing or not
            if missing_ok and not os.path.lexists(path):
                continue
            problem = shard_file_problem(Path(path))
            if problem is not None:
                stored = shard_records.get(serial, "<unknown>")
                problems.append(
                    f'shard #{serial}: stored record "{stored}" resolves to "{str(path)}", which {problem}'
                )

        seen = {}
        for serial, path in sorted(shard_db_files.items()):
            if path in seen:
                problems.append(
                    f'shards #{seen[path]} and #{serial} both resolve to "{str(path)}" (stored records "{shard_records.get(seen[path], "<unknown>")}" and "{shard_records.get(serial, "<unknown>")}")'
                )
            else:
                seen[path] = serial

        return problems

    # COPY, MOVE OR DELETE A CLOSED STORE
    #
    # A store is its primary and its shards, and nothing else: these methods copy, move, delete,
    # check and mention no other file. They are static and work on a closed store. An open pool has
    # one Datastore actor per shard holding its file, so they are not methods of an open pool, and
    # they start no Ray, create no actor and need no instance. They cannot tell whether some process
    # has the store open (a rollback-journal store leaves no file while idle); making sure nothing
    # is using it is the caller's job.
    #
    # copy_store and move_store never delete a file, never overwrite one, and never write the
    # source. delete_store deletes a closed store's own files, and those only: the shards its
    # primary's `shards` table names, read through the one resolver, and then the primary. Its one
    # intended caller is the registry's `store retire`, which keeps the store's sidecar behind as
    # its record. On failure none of them cleans up: each raises, naming the step that failed and
    # the store files that exist. Deleting any other file is for a person.

    @staticmethod
    def copy_store(src: PathType, dst: PathType) -> Dict[int, Path]:
        """
        Copy the closed store whose primary is ``src`` to a new store whose primary is ``dst``,
        renaming every file to ``dst``'s stem, and return the destination's serial -> shard path
        map as read back from the finished destination.

        Refuses, before anything is written, if the source is unusable or not cleanly closed, or
        if any destination name is taken (see _plan_relocation). Then, in this order: each shard
        is copied to its destination name (serial 0 first); the primary is copied to a temporary
        name beside the destination; that copy's `shards` rows are rewritten to the destination's
        bare shard names in one transaction and read back; and it is renamed onto the destination
        primary's name with os.replace. The destination primary therefore appears last and
        complete. Until then the destination has shards and no primary, which the constructor
        refuses to open ("Primary database is missing, but shard ... already exists").

        The source is opened only mode=ro and is never written. No table other than
        `shards.filename` of the destination primary is changed.
        """
        return ShardedPool._relocate_store("copy", src, dst)

    @staticmethod
    def move_store(src: PathType, dst: PathType) -> Dict[int, Path]:
        """
        Move the closed store whose primary is ``src`` so that its primary is ``dst``, renaming
        every file to ``dst``'s stem, and return the destination's serial -> shard path map as
        read back from the finished destination.

        Refuses, before anything is written, under the same conditions as copy_store, and also if
        the destination directory holds a file with the name of one of the source's shards (see
        _plan_relocation). Then, in this order: each shard is renamed to its destination name
        (serial 0 first); the primary is renamed to the destination name; and the destination
        primary's `shards` rows are rewritten to the destination's bare shard names in one
        transaction. Every rename is os.rename. A move across filesystems fails at the first
        rename, before anything has moved; the remedy is to copy the store and then delete the
        source by hand.
        """
        return ShardedPool._relocate_store("move", src, dst)

    @staticmethod
    def closed_store_files(primary: PathType, *, resume: bool = False) -> List[Path]:
        """
        Return every file of the closed store whose primary is ``primary``: its shards in ascending
        serial, then the primary, each absolute. These are exactly the files delete_store with the
        same ``resume`` deletes, in the order it deletes them, because both come from one plan
        (_plan_deletion).

        Refuses exactly as delete_store would, with a RuntimeError naming the file and the reason.
        Under ``resume=True`` a shard whose file is missing is left out of the list rather than
        refused. Nothing is written or deleted: the primary is opened only mode=ro, to read its
        `shards` table.
        """
        return ShardedPool._plan_deletion(primary, resume).files()

    @staticmethod
    def delete_store(primary: PathType, *, resume: bool = False) -> List[Path]:
        """
        Delete the closed store whose primary is ``primary``: each shard its `shards` table names,
        in ascending serial, then the primary. Return the paths deleted, in that order, which are
        closed_store_files(primary, resume=resume).

        The files are named only through _read_closed_store, the one resolver's read-and-check. A
        record that is not a bare file name is refused before anything is deleted, so a primary
        whose records name another store's files deletes nothing, and never those files.

        Refuses, before anything is deleted, if the primary is missing, a symbolic link or not a
        regular file; if the primary or any shard has a -journal, -wal or -shm beside it; if the
        `shards` table cannot be read or records no shard; if any record is unusable or resolves to
        a symbolic link, a non-regular file or a file shared by two serials; if any shard's file is
        missing; or if any file to be deleted is not in the primary's own directory (see
        _plan_deletion). Under ``resume=True`` a missing shard is not refused, and every other
        refusal stands. That completes a deletion that was interrupted, which leaves the primary
        and some of its shards: the primary is deleted last because its `shards` table is the only
        list of the shard files.

        Just before each unlink the path is checked again to be a regular file and not a symbolic
        link. On any failure nothing is cleaned up or retried: it raises, naming the step and every
        file of the store still present. The remedy is delete_store(primary, resume=True), run
        deliberately.
        """
        plan = ShardedPool._plan_deletion(primary, resume)
        steps = [
            (f"delete shard #{serial}", plan.shards[serial])
            for serial in sorted(plan.shards)
        ] + [("delete the primary", plan.primary)]

        deleted: List[Path] = []
        step = "check the files before deleting them"
        try:
            for step, path in steps:
                # the plan checked every file; check again just before the unlink, because a file
                # that changed since is not the file that was planned
                problem = shard_file_problem(path)
                if problem is not None:
                    raise RuntimeError(
                        f'"{str(path)}" {problem}, which it was not when the deletion was planned'
                    )
                os.unlink(path)
                deleted.append(path)
        except Exception as e:
            raise RuntimeError(
                ShardedPool._deletion_failure_message(plan, step, e)
            ) from e

        return deleted

    @staticmethod
    def _journal_paths(path: Path) -> List[Path]:
        """The names SQLite gives the rollback journal and the WAL files of the database ``path``."""
        return [
            path.with_name(path.name + suffix) for suffix in _SQLITE_JOURNAL_SUFFIXES
        ]

    @staticmethod
    def _read_closed_store(
        primary: Path, *, missing_ok: bool = False
    ) -> Tuple[Dict[int, Path], Dict[int, str]]:
        """
        Read the `shards` rows of the closed store ``primary`` (opened mode=ro) and check the files
        they name, through the same two methods the constructor uses. Return (serial -> path,
        serial -> stored record), or raise RuntimeError naming the problem, and nothing else: each
        caller states the operation and the store.
        """
        try:
            conn = sqlite3.connect(f"{primary.as_uri()}?mode=ro", uri=True)
            try:
                rows = conn.execute("SELECT serial, filename FROM shards").fetchall()
            finally:
                conn.close()
        except sqlite3.Error as e:
            raise RuntimeError(f"its shards table could not be read ({e})") from e

        files: Dict[int, Path] = {}
        records: Dict[int, str] = {}
        ShardedPool._resolve_shard_rows(primary, rows, files, records)

        # missing_ok is for delete_store(resume=True) only: a shard file that is not there is not
        # refused. Every serial is still returned, present or not
        problems = ShardedPool._shard_file_problems(
            files, records, missing_ok=missing_ok
        )
        if len(problems) > 0:
            raise RuntimeError("; ".join(problems))

        return files, records

    @staticmethod
    def _plan_relocation(mode: str, src: PathType, dst: PathType) -> "_RelocationPlan":
        """
        Every refusal of copy_store and move_store, made before anything is written. Returns what
        the operation will do. Raises RuntimeError naming the file and the reason.
        """

        def refuse(reason: str) -> RuntimeError:
            return RuntimeError(
                f'Cannot {mode} sharded datastore "{str(src)}" to "{str(dst)}": {reason}. Nothing was written'
            )

        # the source primary: an existing regular file, not a symbolic link (the same test as a
        # shard), and cleanly closed. Its journal is checked before it is opened
        src_given = Path(src).absolute()
        problem = shard_file_problem(src_given)
        if problem is not None:
            raise refuse(f'the source primary "{str(src_given)}" {problem}')
        src_primary = src_given.resolve()

        def refuse_journals(path: Path, what: str):
            for journal in ShardedPool._journal_paths(path):
                if os.path.lexists(journal):
                    raise refuse(
                        f'{what} "{str(path)}" has "{str(journal)}" beside it, so it was not closed cleanly (a copy made without that file is corrupt)'
                    )

        refuse_journals(src_primary, "the source primary")

        # the source's shard records, read and checked exactly as the constructor reads them
        try:
            src_files, src_records = ShardedPool._read_closed_store(src_primary)
        except RuntimeError as e:
            raise refuse(str(e)) from e

        if len(src_files) == 0:
            raise refuse(f'the source primary "{str(src_primary)}" records no shards')
        # an interrupted operation leaves destination shards and no destination primary, which
        # the constructor refuses because shard 0 exists. That needs shard 0 to be there, and to
        # be the first one written
        if 0 not in src_files:
            raise refuse(
                f'the source primary "{str(src_primary)}" records no shard #0 (serials {sorted(src_files)}), so an interrupted {mode} could not be told apart from an unused name'
            )

        for serial, path in sorted(src_files.items()):
            refuse_journals(Path(path), f"source shard #{serial}")

        # the destination primary: a new file, and not the source
        dst_given = Path(dst)
        if dst_given.is_dir():
            raise refuse(
                f'the destination "{str(dst_given.absolute())}" is an existing directory; the destination names the new primary file'
            )
        dst_primary = dst_given.resolve()
        if dst_primary == src_primary or (
            os.path.lexists(dst_given) and os.path.samefile(dst_given, src_primary)
        ):
            raise refuse(
                f'the destination "{str(dst_primary)}" is the same file as the source'
            )

        # the destination's shard names come from the one naming rule, for the serials in the
        # source's table
        dst_dir = dst_primary.parent
        dst_records = {
            serial: shard_file_name(dst_primary, serial) for serial in src_files
        }
        dst_files = {serial: dst_dir / name for serial, name in dst_records.items()}
        temp_primary = (
            dst_primary.with_name(dst_primary.name + _INCOMPLETE_COPY_SUFFIX)
            if mode == "copy"
            else None
        )

        # nothing is ever overwritten, and a stale journal at a destination name would be replayed
        # into the new file by its next opener, so none of those names may exist either
        taken = []
        for path in [dst_primary, *[dst_files[s] for s in sorted(dst_files)]] + (
            [temp_primary] if temp_primary is not None else []
        ):
            for name in [path, *ShardedPool._journal_paths(path)]:
                if os.path.lexists(name):
                    taken.append(f'"{str(name)}"')
        if len(taken) > 0:
            raise refuse(
                "the destination name(s) " + ", ".join(taken) + " already exist"
            )

        # a move renames the primary before rewriting its rows, so for a moment the destination
        # primary still holds the source's records, which it reads by name in its new directory.
        # Where they would find files that are not this store's, refuse
        if mode == "move" and dst_dir != src_primary.parent:
            for serial, path in sorted(src_files.items()):
                other = dst_dir / Path(path).name
                if os.path.lexists(other):
                    raise refuse(
                        f'the destination directory holds "{str(other)}", which has the name of source shard #{serial}; a move interrupted before its rows were rewritten would read that file as shard #{serial}'
                    )

        return _RelocationPlan(
            mode=mode,
            src_primary=src_primary,
            src_files={s: Path(p) for s, p in src_files.items()},
            dst_primary=dst_primary,
            dst_files=dst_files,
            dst_records=dst_records,
            temp_primary=temp_primary,
        )

    @staticmethod
    def _write_shard_records(primary: Path, records: Dict[int, str]) -> None:
        """
        Set `shards.filename` of the existing primary ``primary`` to ``records`` (serial -> bare
        name), in one transaction, changing nothing else. The table must hold exactly those serials.
        """
        conn = sqlite3.connect(f"{primary.as_uri()}?mode=rw", uri=True)
        try:
            with conn:
                for serial, name in sorted(records.items()):
                    cursor = conn.execute(
                        "UPDATE shards SET filename = ? WHERE serial = ?",
                        (name, serial),
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError(
                            f'the shards table of "{str(primary)}" has {cursor.rowcount} rows for serial #{serial}, expected 1'
                        )
                (count,) = conn.execute("SELECT COUNT(*) FROM shards").fetchone()
                if count != len(records):
                    raise RuntimeError(
                        f'the shards table of "{str(primary)}" has {count} rows, expected {len(records)}'
                    )
        finally:
            conn.close()

    @staticmethod
    def _read_back(plan: "_RelocationPlan", primary: Path) -> Dict[int, Path]:
        """Read ``primary``'s records back through the constructor's read-and-check, and require
        them to be exactly the destination's bare names, resolving to the destination's files.
        """
        files, records = ShardedPool._read_closed_store(primary)
        if records != plan.dst_records or files != plan.dst_files:
            raise RuntimeError(
                f'"{str(primary)}" reads back records {records} resolving to {({s: str(p) for s, p in files.items()})}, expected {plan.dst_records}'
            )
        return dict(sorted(files.items()))

    @staticmethod
    def _relocate_store(mode: str, src: PathType, dst: PathType) -> Dict[int, Path]:
        plan = ShardedPool._plan_relocation(mode, src, dst)
        serials = sorted(plan.dst_files)  # serial 0 first: see _plan_relocation

        def no_overwrite(path: Path):
            # checked once for every name in _plan_relocation; checked again just before each
            # write, because shutil.copy2 and os.rename both replace an existing file silently
            if os.path.lexists(path):
                raise FileExistsError(
                    errno.EEXIST, "refusing to overwrite an existing file", str(path)
                )

        step = "create the destination directory"
        try:
            plan.dst_primary.parent.mkdir(parents=True, exist_ok=True)

            if mode == "copy":
                for serial in serials:
                    step = f"copy shard #{serial}"
                    no_overwrite(plan.dst_files[serial])
                    shutil.copy2(plan.src_files[serial], plan.dst_files[serial])

                step = "copy the primary to its temporary name"
                no_overwrite(plan.temp_primary)
                shutil.copy2(plan.src_primary, plan.temp_primary)

                step = "rewrite the temporary primary's shards rows"
                ShardedPool._write_shard_records(plan.temp_primary, plan.dst_records)

                step = "read back the temporary primary"
                ShardedPool._read_back(plan, plan.temp_primary)

                step = "rename the temporary primary to the destination primary"
                no_overwrite(plan.dst_primary)
                os.replace(plan.temp_primary, plan.dst_primary)

            else:
                for serial in serials:
                    step = f"rename shard #{serial}"
                    no_overwrite(plan.dst_files[serial])
                    os.rename(plan.src_files[serial], plan.dst_files[serial])

                step = "rename the primary"
                no_overwrite(plan.dst_primary)
                os.rename(plan.src_primary, plan.dst_primary)

                step = "rewrite the destination primary's shards rows"
                ShardedPool._write_shard_records(plan.dst_primary, plan.dst_records)

            step = "read back the destination"
            return ShardedPool._read_back(plan, plan.dst_primary)

        except Exception as e:
            raise RuntimeError(ShardedPool._failure_message(plan, step, e)) from e

    @staticmethod
    def _failure_message(plan: "_RelocationPlan", step: str, e: Exception) -> str:
        def existing(paths) -> str:
            names = [
                f'"{str(name)}"'
                for path in paths
                for name in [path, *ShardedPool._journal_paths(path)]
                if os.path.lexists(name)
            ]
            return "[" + ", ".join(names) + "]"

        dst_paths = [
            plan.dst_primary,
            *[plan.dst_files[s] for s in sorted(plan.dst_files)],
        ]
        if plan.temp_primary is not None:
            dst_paths.append(plan.temp_primary)
        src_paths = [
            plan.src_primary,
            *[plan.src_files[s] for s in sorted(plan.src_files)],
        ]

        message = (
            f'{plan.mode} of sharded datastore "{str(plan.src_primary)}" to "{str(plan.dst_primary)}" failed at step "{step}": {type(e).__name__}: {e}. '
            f"Nothing has been deleted or cleaned up; that is for a person. "
            f"Store files now at the destination: {existing(dst_paths)}; at the source: {existing(src_paths)}"
        )
        if isinstance(e, OSError) and e.errno == errno.EXDEV:
            message += (
                ". The source and the destination are on different filesystems, so the store "
                "cannot be moved by renaming it; copy it instead, and then delete the source by hand"
            )
        return message

    @staticmethod
    def _plan_deletion(primary: PathType, resume: bool) -> "_DeletionPlan":
        """
        Every refusal of closed_store_files and delete_store, made before anything is deleted, and
        the one list of the files both name. Raises RuntimeError naming the file and the reason.

        The shard files come from _read_closed_store and from nothing else: the stored records are
        read only through the one resolver, and never opened as paths. The primary is opened only
        mode=ro, and nothing is written.
        """
        given = Path(primary).absolute()

        def refuse(reason: str) -> RuntimeError:
            return RuntimeError(
                f'Cannot delete sharded datastore "{str(given)}": {reason}. Nothing was deleted'
            )

        # the primary: an existing regular file, not a symbolic link (the same test as a shard, as
        # _plan_relocation makes of a source). Without it nothing names the store's shards, so
        # nothing is looked for by any other rule
        problem = shard_file_problem(given)
        if problem is not None:
            reason = f'the primary "{str(given)}" {problem}'
            if not os.path.lexists(given):
                reason += (
                    "; its shards table is the only record of which files are this store's, so "
                    "with no primary no file can be identified as the store's"
                )
            raise refuse(reason)
        primary_file = given.resolve()

        def refuse_journals(path: Path, what: str):
            for journal in ShardedPool._journal_paths(path):
                if os.path.lexists(journal):
                    raise refuse(
                        f'{what} "{str(path)}" has "{str(journal)}" beside it, so it was not closed cleanly or is open now'
                    )

        # checked before the primary is opened
        refuse_journals(primary_file, "the primary")

        # the shard records, read and checked exactly as the constructor reads them. Under resume a
        # missing shard is not refused, and nothing else is relaxed
        try:
            recorded, _records = ShardedPool._read_closed_store(
                primary_file, missing_ok=resume
            )
        except RuntimeError as e:
            raise refuse(str(e)) from e

        if len(recorded) == 0:
            raise refuse(f'the primary "{str(primary_file)}" records no shards')

        # the resolver puts every shard in the primary's directory. Asserted here as well, so that
        # no later change to the resolver can make this method delete a file anywhere else
        directory = primary_file.parent
        for serial, path in sorted(recorded.items()):
            if Path(path).parent != directory:
                raise refuse(
                    f'shard #{serial} is "{str(path)}", in the directory "{str(Path(path).parent)}", which is not the primary\'s own directory "{str(directory)}"'
                )

        for serial, path in sorted(recorded.items()):
            refuse_journals(Path(path), f"shard #{serial}")

        # every shard, or under resume those still present, in ascending serial
        shards = {
            serial: Path(path)
            for serial, path in sorted(recorded.items())
            if not resume or os.path.lexists(path)
        }

        return _DeletionPlan(
            primary=primary_file,
            recorded={s: Path(p) for s, p in sorted(recorded.items())},
            shards=shards,
        )

    @staticmethod
    def _deletion_failure_message(
        plan: "_DeletionPlan", step: str, e: Exception
    ) -> str:
        names = [
            f'"{str(name)}"'
            for path in [
                *[plan.recorded[s] for s in sorted(plan.recorded)],
                plan.primary,
            ]
            for name in [path, *ShardedPool._journal_paths(path)]
            if os.path.lexists(name)
        ]
        return (
            f'delete of sharded datastore "{str(plan.primary)}" failed at step "{step}": {type(e).__name__}: {e}. '
            f"Nothing has been cleaned up, and the deletion was not retried. "
            f"Store files still present: [{', '.join(names)}]. "
            f'Once the failure is understood, complete the deletion deliberately with ShardedPool.delete_store("{str(plan.primary)}", resume=True)'
        )

    def object_get(self, ObjectClass, **kwargs):
        if isinstance(ObjectClass, str):
            cls_name = ObjectClass
        else:
            cls_name = ObjectClass.__name__

        if cls_name in self._replicated_tables:
            return self._get_impl_replicated_table(cls_name, kwargs)

        if cls_name in self._sharded_tables.keys():
            return self._get_impl_sharded_table(cls_name, kwargs)

        raise RuntimeError(
            f'Unable to dispatch object_get() for item of type "{cls_name}"'
        )

    # THE REPLICATED WRITE
    #
    # A replicated object is written on a controlling shard, chosen at random per call, and then
    # sent to every other shard, each in its own transaction. _replicated_write is the one path
    # the get, store and validate of a replicated class take, the version row's included
    # (ShardedPool.__init__ writes it here, prompts/a3-v2-readiness prompt 02), and the one place
    # that writes and clears the primary's replication_in_flight record for them
    # (prompts/datastore-integrity, prompt 01). The one other writer of that record is the prune
    # at open of a replicated class, whose "prune" record _prune_replicated_tables writes and
    # clears, with no Ray, before any actor exists:
    #
    #   1. the record is committed on the primary, in its own transaction, naming the operation,
    #      the class and the controlling shard. If one is already there, nothing is written;
    #   2. the controlling shard's call is submitted and its answer read;
    #   3. if a replication follows, the record's store_id is filled in (when there is one) and
    #      every replica's call is submitted, each with the controller's serial;
    #   4. every replica's answer is checked against the controller's; a difference raises
    #      ReplicationMismatch;
    #   5. only then is the record deleted.
    #
    # Any exception after step 1 leaves the record set, whether or not the shards in fact differ:
    # a failure reported by the controlling shard may have followed its commit (an actor that dies
    # after committing), and the record is what says so. One timestamp, chosen here, is passed to
    # every shard, so that copies of a row are identical.

    def _replicated_write(
        self,
        operation: str,
        cls_name: str,
        submit_controller: Callable,
        replication: Callable,
        store_id: Optional[int] = None,
    ):
        """
        Run one replicated write, and return (the controlling shard's ref, its answer).

        ``submit_controller(handle, timestamp)`` submits the controlling shard's call and returns
        its ref. ``replication(controller_shard_id, answer)`` returns None if nothing is to be
        replicated, or a _Replication saying how. ``store_id`` is written into the record at once
        when it is known before the controller's call (a validate).
        """
        if getattr(self, "_replication_active", False):
            # replicated calls never overlap in the driver (audit §4.1); this makes it an assertion
            raise RuntimeError(
                f'ShardedPool: a replicated {operation} of "{cls_name}" was started while another '
                "replicated write was in progress. Replicated writes must not overlap"
            )
        self._replication_active = True
        try:
            # pick a shard id at random to be the "controlling" shard: swap this entry with the
            # last element, then pop it
            shard_ids = list(self._shards.keys())
            i = random.randrange(len(shard_ids))
            shard_ids[i], shard_ids[-1] = shard_ids[-1], shard_ids[i]
            controller = shard_ids.pop()

            # the one timestamp of this write, and the record's start time
            timestamp = datetime.now()

            record_table = self._replication_in_flight_table

            # 1. the record, committed before the controlling shard's call is submitted
            with self._engine.begin() as conn:
                existing = conn.execute(sqla.select(record_table)).mappings().first()
                if existing is not None:
                    raise ReplicationInFlight(self._primary_file, dict(existing))
                conn.execute(
                    sqla.insert(record_table),
                    {
                        "operation": operation,
                        "class_name": cls_name,
                        "controller_shard": controller,
                        "store_id": store_id,
                        "started": timestamp,
                    },
                )

            # 2. the controlling shard
            ref = submit_controller(self._shards[controller], timestamp)
            answer = ray.get(ref)

            # 3. the replicas, if anything is to be replicated
            plan = replication(controller, answer)
            if plan is not None:
                if plan.store_id is not None and plan.store_id != store_id:
                    with self._engine.begin() as conn:
                        result = conn.execute(
                            sqla.update(record_table).values(store_id=plan.store_id)
                        )
                        if result.rowcount != 1:
                            raise RuntimeError(
                                f"ShardedPool: the replication_in_flight record of a replicated "
                                f'{operation} of "{cls_name}" has {result.rowcount} rows, expected 1'
                            )

                # every replica's call is submitted before any answer is read
                replica_refs = [
                    (key, plan.submit(self._shards[key], timestamp))
                    for key in shard_ids
                ]
                replica_answers = ray.get([r for _, r in replica_refs])

                # 4. every answer checked against the controller's, before the record is cleared
                for (key, _), replica_answer in zip(replica_refs, replica_answers):
                    plan.check(key, replica_answer)

            # 5. every shard holds what the controller holds: clear the record
            with self._engine.begin() as conn:
                result = conn.execute(sqla.delete(record_table))
                if result.rowcount != 1:
                    raise RuntimeError(
                        f"ShardedPool: the replication_in_flight record of a replicated "
                        f'{operation} of "{cls_name}" has {result.rowcount} rows, expected 1'
                    )

            return ref, answer

        finally:
            self._replication_active = False

    @staticmethod
    def _needs_replication(obj) -> bool:
        """A get's answer is replicated if it has a serial and was inserted or updated."""
        return (
            hasattr(obj, "_my_id")
            and obj._my_id is not None
            and (hasattr(obj, "_new_insert") or hasattr(obj, "_updated"))
        )

    def _get_impl_replicated_table(self, cls_name, kwargs):
        # a read-only pool answers from one shard, with no record and no replication (prompt 03)
        if getattr(self, "_read_only", False):
            return self._get_impl_replicated_table_read_only(cls_name, kwargs)

        # we push an initial 'get' to the controlling shard. If a new database object was created
        # (or an existing one updated) by the get, we then push a replica to all the other shards,
        # with the controller's store_id, and require each to answer with that store_id.
        # For replicated tables there is no need to use our internal information about the
        # next-allocated store_id, and in fact doing so would make the logic here much more
        # complicated. So we avoid that.
        def submit_controller(handle, timestamp):
            return handle.object_get.remote(
                cls_name, insert_timestamp=timestamp, **kwargs
            )

        def replication(controller, objects):
            # was this a vectorized get?
            if "payload_data" in kwargs:
                payload_data = kwargs["payload_data"]

                if len(payload_data) != len(objects):
                    raise RuntimeError(
                        f"object_get() data returned from selected datastore (shared={controller}) has a different length (length={len(objects)}) to payload data (length={len(payload_data)})"
                    )

                # add explicit serial specifier to each object that must be replicated
                new_payload = []
                expected = []
                for i in range(len(payload_data)):
                    if ShardedPool._needs_replication(objects[i]):
                        payload_data[i]["serial"] = objects[i].store_id
                        new_payload.append(payload_data[i])
                        expected.append(objects[i].store_id)

                # a get that inserts nothing needs no replication
                if len(new_payload) == 0:
                    return None

                def submit(handle, timestamp):
                    return handle.object_get.remote(
                        cls_name, insert_timestamp=timestamp, payload_data=new_payload
                    )

                def check(shard_id, answers):
                    returned = [getattr(a, "_my_id", None) for a in answers]
                    if len(returned) != len(expected):
                        raise ReplicationMismatch(
                            shard_id,
                            cls_name,
                            expected,
                            returned,
                            detail=f"the replica returned {len(returned)} objects for {len(expected)}",
                        )
                    for want, got in zip(expected, returned):
                        if got != want:
                            raise ReplicationMismatch(shard_id, cls_name, want, got)

                return _Replication(store_id=None, submit=submit, check=check)

            # this was a scalar get
            if not ShardedPool._needs_replication(objects):
                return None

            serial = objects.store_id

            def submit(handle, timestamp):
                return handle.object_get.remote(
                    cls_name, serial=serial, insert_timestamp=timestamp, **kwargs
                )

            def check(shard_id, answer):
                got = getattr(answer, "_my_id", None)
                if got != serial:
                    raise ReplicationMismatch(shard_id, cls_name, serial, got)

            return _Replication(store_id=serial, submit=submit, check=check)

        ref, objects = self._replicated_write(
            "get", cls_name, submit_controller, replication
        )

        # test whether this query was for a shard key, and, if so, assign any shard keys
        # that are missing
        if cls_name == self._ShardKeyType_name:
            self._assign_shard_keys(objects)

        # return the controlling shard's object; every replica's answer has been checked against it
        return ref

    def _get_impl_sharded_table(self, cls_name, kwargs):
        # for sharded tables, we should query/insert into only the appropriate shard
        shard_key_field = self._sharded_tables[cls_name]

        # is this a vectorized get?
        if "payload_data" in kwargs:
            payload_data = kwargs["payload_data"]

            work_refs = []
            for item in payload_data:
                key = item[shard_key_field]
                shard_id = self._shard_keys[self._ShardKeyStoreIdGetter(key)]

                work_refs.append(
                    self._shards[shard_id].object_get.remote(cls_name, **item)
                )

            return work_refs
            # TODO: consider consolidating all objects for the same shard into a list, for efficiency

        # otherwise, can assume this is scalar get
        key = kwargs[shard_key_field]
        shard_id = self._shard_keys[self._ShardKeyStoreIdGetter(key)]

        return self._shards[shard_id].object_get.remote(cls_name, **kwargs)

    def object_get_vectorized(self, ObjectClass, shard_key, payload_data):
        if isinstance(ObjectClass, str):
            cls_name = ObjectClass
        else:
            cls_name = ObjectClass.__name__

        if cls_name not in self._sharded_tables:
            raise RuntimeError(
                f"ShardedPool: it is only possible to vectorize object_get() over a sharded table (object type={cls_name})"
            )

        shard_key_field = self._sharded_tables[cls_name]
        if shard_key_field not in shard_key:
            raise RuntimeError(
                f'ShardedPool: expected shard key "{shard_key_field}" to be provided for object type "{cls_name}", but instead received keys: {shard_key.keys()}'
            )

        shard_id = self._shard_keys[
            self._ShardKeyStoreIdGetter(shard_key[shard_key_field])
        ]

        for value in payload_data:
            value.update(shard_key)
        return self._shards[shard_id].object_get.remote(
            cls_name, payload_data=payload_data
        )

    def object_read_batch(self, ObjectClass, shard_key, **payload):
        if isinstance(ObjectClass, str):
            cls_name = ObjectClass
        else:
            cls_name = ObjectClass.__name__

        if cls_name not in self._sharded_tables:
            raise RuntimeError(
                f"ShardedPool: it is only possible to apply object_read_batch() to a sharded table (object type={cls_name})"
            )

        shard_key_field = self._sharded_tables[cls_name]
        if shard_key_field not in shard_key:
            raise RuntimeError(
                f'ShardedPool: expected shard key "{shard_key_field}" to be provided for object type "{cls_name}", but instead received keys: {shard_key.keys()}'
            )

        shard_id = self._shard_keys[
            self._ShardKeyStoreIdGetter(shard_key[shard_key_field])
        ]

        payload.update(shard_key)
        return self._shards[shard_id].object_read_batch.remote(cls_name, **payload)

    def object_store(self, objects):
        # a read-only pool stores nothing (prompts/a3-v2-readiness, prompt 03)
        if getattr(self, "_read_only", False):
            self._refuse_write("object_store", objects)

        if isinstance(objects, list) or isinstance(objects, tuple):
            payload_data = objects
            scalar = False
        else:
            payload_data = [objects]
            scalar = True

        work_refs = []
        for item in payload_data:
            cls_name = type(item).__name__

            if cls_name in self._replicated_tables:
                work_refs.extend(self._store_impl_replicated_table(cls_name, item))
                continue

            if cls_name in self._sharded_tables.keys():
                work_refs.extend(self._store_impl_sharded_table(cls_name, item))
                continue

            raise RuntimeError(
                f'Unable to dispatch object_get() for item of type "{cls_name}"'
            )

        if scalar:
            return work_refs[0]

        return work_refs

    def _store_impl_replicated_table(self, cls_name, item):
        # we push an initial 'store' to the controlling shard, and then push the object, complete
        # with its new store_id (and the serials of the rows it owns, which its factory's
        # owned_serials gives), to all the other shards, each of which inserts it under those
        # serials or verifies that it already holds it
        def submit_controller(handle, timestamp):
            return handle.object_store.remote(item, insert_timestamp=timestamp)

        def replication(controller, obj):
            if not hasattr(obj, "_my_id") or obj._my_id is None:
                raise RuntimeError(
                    f'Stored object of type "{cls_name}" was not assigned a store_id field'
                )

            serial = obj._my_id
            factory = self._factories[cls_name]
            value_serials = factory.owned_serials(obj)

            def submit(handle, timestamp):
                return handle.object_store.remote(obj, insert_timestamp=timestamp)

            def check(shard_id, answer):
                got = getattr(answer, "_my_id", None)
                if got != serial:
                    raise ReplicationMismatch(shard_id, cls_name, serial, got)
                got_values = factory.owned_serials(answer)
                if got_values != value_serials:
                    raise ReplicationMismatch(
                        shard_id,
                        cls_name,
                        serial,
                        got,
                        detail=f"its value rows have serials {got_values}, the controller's {value_serials}",
                    )

            return _Replication(store_id=serial, submit=submit, check=check)

        ref, _ = self._replicated_write(
            "store", cls_name, submit_controller, replication
        )
        return [ref]

    def _store_impl_sharded_table(self, cls_name, item):
        # item need only be pushed to a single shard
        # unlike the replicated case,
        # we don't have to care about what happens to its store_id

        shard_key_field = self._sharded_tables[cls_name]
        if not hasattr(item, shard_key_field):
            raise RuntimeError(
                f'Unable to determine shard, because object of type "{cls_name}" has no "{shard_key_field}" attribute'
            )

        key = getattr(item, shard_key_field)
        shard_id = self._shard_keys[self._ShardKeyStoreIdGetter(key)]

        # TODO: consider consolidating all stores for the same shard into a list, for efficiency
        return [self._shards[shard_id].object_store.remote(item)]

    def object_validate(self, objects):
        # a read-only pool validates nothing (prompts/a3-v2-readiness, prompt 03)
        if getattr(self, "_read_only", False):
            self._refuse_write("object_validate", objects)

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

            if cls_name in self._replicated_tables:
                work_refs.extend(self._validate_impl_replicated_table(cls_name, item))
                continue

            if cls_name in self._sharded_tables.keys():
                work_refs.extend(self._validate_impl_sharded_table(cls_name, item))
                continue

            raise RuntimeError(
                f'Unable to dispatch object_validate() for item of type "{cls_name}"'
            )

        if scalar:
            return work_refs[0]

        return work_refs

    def _validate_impl_replicated_table(self, cls_name, item):
        # we push an initial 'validate' to the controlling shard. If it validates, every other
        # shard validates its own copy, and every outcome must equal the controller's
        serial = getattr(item, "_my_id", None)

        def submit_controller(handle, timestamp):
            return handle.object_validate.remote(item)

        def replication(controller, outcome):
            # if object did not validate, do not push validation requests to remaining shards
            if outcome is False or outcome is None:
                return None

            def submit(handle, timestamp):
                return handle.object_validate.remote(item)

            def check(shard_id, replica_outcome):
                if replica_outcome != outcome:
                    print(f"!! Validation outcomes did not agree between shards:")
                    print(
                        f"|    controller (shard {controller}) = {outcome}, shard {shard_id} = {replica_outcome}"
                    )
                    raise ReplicationMismatch(
                        shard_id,
                        cls_name,
                        serial,
                        serial,
                        detail=f"its validation outcome is {replica_outcome}, the controller's {outcome}",
                    )

            return _Replication(store_id=serial, submit=submit, check=check)

        ref, _ = self._replicated_write(
            "validate", cls_name, submit_controller, replication, store_id=serial
        )
        return [ref]

    def _validate_impl_sharded_table(self, cls_name, item):
        # item need only be validated on a single shard
        shard_key_field = self._sharded_tables[cls_name]
        if not hasattr(item, shard_key_field):
            raise RuntimeError(
                f'Unable to determine shard, because object of type "{cls_name}" has no "{shard_key_field}" attribute'
            )

        key = getattr(item, shard_key_field)
        shard_id = self._shard_keys[self._ShardKeyStoreIdGetter(key)]

        # TODO: consider consolidating all validates for the same shard into a list, for efficiency
        return [self._shards[shard_id].object_validate.remote(item)]

    def _assign_shard_keys(self, obj):
        if isinstance(obj, list):
            data = obj
        else:
            data = [obj]

        # assign any shard keys that we can, without going out to the database
        # (because this is bound to be slower)
        seen_store_ids = set()
        missing_keys = []
        for item in data:
            if not isinstance(item, self._ShardKeyType):
                raise RuntimeError(
                    f'shard keys should be of type "{self._ShardKeyType_name}"'
                )

            if (
                item.store_id not in self._shard_keys
                and item.store_id not in seen_store_ids
            ):
                missing_keys.append(item)
                seen_store_ids.add(item.store_id)

        # if no work to do, return
        if len(missing_keys) == 0:
            return

        # a read-only pool assigns no shard key (prompts/a3-v2-readiness, prompt 03): the check at
        # open compared shard_keys with the rows of the shard-key class every shard holds, so a
        # row with no key here is one the store's primary does not know
        if getattr(self, "_read_only", False):
            raise ReadOnlyWrite(
                self._primary_file,
                "assigning a shard key",
                class_name=self._ShardKeyType_name,
                detail=f"store_id(s) {sorted(seen_store_ids)} have no row in the primary's "
                "shard_keys",
            )

        # otherwise, we have to populate keys
        # try to load balance by working out which shard has the fewest keys
        loads = {key: 0 for key in self._shards.keys()}
        for shard in self._shard_keys.values():
            loads[shard] = loads[shard] + 1

        with self._engine.begin() as conn:
            for item in missing_keys:
                # find which shard has the current minimum load
                if len(loads) > 0:
                    new_shard = min(loads, key=loads.get)
                else:
                    new_shard = list(self._shards.keys()).pop()

                # insert a new record for this key
                result = conn.execute(
                    sqla.insert(self._shard_key_table),
                    {"key_serial": item.store_id, "shard_id": new_shard},
                )
                assigned_serial = result.inserted_primary_key[0]

                if assigned_serial != item.store_id:
                    print(
                        f"!! _assign_shard_keys MISMATCH: "
                        f"store_id={item.store_id}, "
                        f"assigned key_serial={assigned_serial}, "
                        f"shard={new_shard}"
                    )

                self._shard_keys[item.store_id] = new_shard
                loads[new_shard] = loads[new_shard] + 1

                # print(
                #     f">> assigned shard #{new_shard} to key object #{item.store_id}"
                # )

            conn.commit()

    def read_table(self, cls, *args, **kwargs):
        """
        Provide a generic service to read a replicated table using an underlying Datastore
        :param cls:
        :param args:
        :param kwargs:
        :return:
        """
        if self._read_table_config is None:
            raise RuntimeError("ShardedPool: the read_table service is not configured")

        if isinstance(cls, str):
            class_name = cls
        else:
            class_name = cls.__name__

        if class_name in self._sharded_tables:
            raise RuntimeError(
                f'ShardedPool: the read_table service is only available for replicated tables, but "{class_name}" is configured as a sharded table'
            )

        if class_name not in self._read_table_config:
            raise RuntimeError(
                f'ShardedPool: the read_table service is not available for objects of class "{class_name}"'
            )

        # we only need to read the table from a single shard, so pick one at random
        shard_ids = list(self._shards.keys())
        i = random.randrange(len(shard_ids))

        # swap this entry with the last element, then pop it
        shard_ids[i], shard_ids[-1] = shard_ids[-1], shard_ids[i]
        shard_key = shard_ids.pop()

        shard = self._shards[shard_key]

        return shard.read_table.remote(class_name, *args, **kwargs)
