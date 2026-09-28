import errno
import os
import random
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Callable, NamedTuple, Tuple

import ray
import sqlalchemy as sqla

from Datastore.SQL import Datastore
from Datastore.SQL.Datastore import PathType, ReadTableConfigType, _factories
from Datastore.SQL.ProfileAgent import ProfileAgent
from Datastore.SQL.SerialPoolBroker import SerialPoolBroker
from Datastore.SQL.schema import build_schema
from Datastore.object import DatastoreObject
from Datastore.replication import (
    ReplicatedDivergence,
    ReplicationInFlight,
    ReplicationMismatch,
    same_value,
)
from Datastore.shard_paths import (
    resolve_shard_path,
    is_legacy_record,
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
        drop_actions: Optional[List[str]] = None,
        read_table_config: Optional[ReadTableConfigType] = None,
    ) -> None:
        """
        Initialize a pool of datastore actors
        :param replicated_tables:
        :param sharded_tables:
        :param ShardKeyStoreIdGetter:
        :param version_label:
        """
        self._job_name: Optional[str] = job_name
        self._version_label: str = version_label

        self._prune_unvalidated: Optional[List[str]] = prune_unvalidated

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
            self._read_shard_data()

            # fail closed, before any actor exists: a Datastore actor given a missing shard file
            # creates an empty database there, and the pool would open with nothing in it
            self._check_shard_files()

            # before any actor exists, and with no Ray: roll back hot journals, compare every
            # replicated table across the shards, repair an interrupted replication from its
            # record's controlling shard, and refuse to open on any other difference
            self.reconciliation = self._reconcile_replicated_tables(primary_journal)

            # a store written before the replicated write was recorded has no in-flight table;
            # create it, empty, only once the store has been checked
            self._replication_in_flight_table.create(self._engine, checkfirst=True)

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
            profile_agent=self._profile_agent,
            prune_unvalidated=self._prune_unvalidated,
            drop_actions=drop_actions,
            read_table_config=read_table_config,
        )
        self._shards = {shard0_key: shard0_store}

        # get the version label from this store
        self._version = ray.get(
            shard0_store.object_get.remote("version", label=version_label)
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
                    profile_agent=self._profile_agent,
                    prune_unvalidated=self._prune_unvalidated,
                    drop_actions=drop_actions,
                    read_table_config=read_table_config,
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

        self._read_table_config: Optional[ReadTableConfigType] = read_table_config
        if read_table_config is not None:
            for class_name, config in read_table_config.items():
                if class_name not in self._replicated_tables:
                    raise RuntimeError(
                        f'It is only possible to configure a read-table method for a replicated table (class name="{class_name}")'
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
        # at most one row, naming the replicated write in progress: committed before the
        # controlling shard's call is submitted and deleted after the last replica's call has
        # returned and been checked (_replicated_write). A row here means the shards may differ,
        # and that the controlling shard's rows of this class are the right ones
        self._replication_in_flight_table = sqla.Table(
            "replication_in_flight",
            self._metadata,
            sqla.Column(
                "operation", sqla.String(DEFAULT_STRING_LENGTH), nullable=False
            ),
            sqla.Column(
                "class_name", sqla.String(DEFAULT_STRING_LENGTH), nullable=False
            ),
            sqla.Column("controller_shard", sqla.Integer, nullable=False),
            sqla.Column("store_id", sqla.Integer, nullable=True),
            sqla.Column("started", sqla.DateTime(), nullable=False),
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
    # in the write path deletes a replicated row, the controller commits first and the replicas
    # copy only what it committed, and a flag is only ever turned on, so after an interrupted
    # replication the right state is exactly "every shard holds what the controller holds"
    # (docs/datastore-integrity-audit.md §4.5). _reconcile_replicated_tables runs in the
    # constructor after _check_shard_files and before the broker or any actor exists, on plain
    # SQLAlchemy connections to the shard files through tables built by build_schema, with no Ray:
    #
    #   1. each shard is opened read-write once, so that SQLite rolls back a hot journal;
    #   2. every replicated table, and the tag table of every replicated class, is read from every
    #      shard and compared by full row; the primary's shard_keys are checked against the
    #      wavenumbers every shard holds;
    #   3. no record and no difference: nothing is written, and the store opens as before;
    #   4. a record, and differences its interrupted write explains: each row of the record's
    #      class that the record's controlling shard holds and another shard lacks is copied to
    #      that shard verbatim (a BackgroundModel as a unit: row, tags and values), a flag the
    #      controller holds set is set, and after a validate BackgroundModel.validated is
    #      recomputed on each shard by the factory's own rule. Every shard is then compared again,
    #      inside the repair's transactions; only if they all agree are the shards committed, and
    #      then the record is cleared, in its own commit, last. A record over identical shards is
    #      simply cleared;
    #   5. anything else raises ReplicatedDivergence, and nothing is written.
    #
    # This deletes no row but the record's own and sets no flag to False. Nothing in a sharded
    # table is read, repaired or refused on: those are not replicated, and nothing here can say
    # what they should be.

    def _reconcile_replicated_tables(self, primary_journal: bool = False) -> dict:
        """
        Compare the replicated tables of this existing store across its shards, repair an
        interrupted replication named by the primary's replication_in_flight record, and raise
        ReplicatedDivergence on any other difference. Return the reconciliation record: the
        tables compared, the files that had a hot journal, the in-flight record found (if any)
        and its controlling shard, what was repaired, and whether the record was cleared.
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

        built = build_schema(sqla.MetaData(), _factories)
        specs = self._replicated_table_specs(built)
        reconciliation["compared"] = [spec["name"] for spec in specs]

        record = self._read_in_flight_record()
        reconciliation["record"] = record
        if record is not None:
            reconciliation["controller"] = record.get("controller_shard")

        engines = {
            sid: self._reconcile_engine(self._shard_db_files[sid]) for sid in shard_ids
        }
        conns = {}
        try:
            for sid in shard_ids:
                conns[sid] = engines[sid].connect()

            # 2. the comparison
            snapshots = {
                sid: self._read_replicated_tables(conns[sid], specs)
                for sid in shard_ids
            }
            for conn in conns.values():
                conn.rollback()
            differences = self._replicated_differences(specs, snapshots)

            # 3. no record: any difference refuses
            if record is None:
                if len(differences) > 0:
                    raise ReplicatedDivergence(
                        self._primary_file,
                        self._describe_differences(specs, snapshots, differences),
                        None,
                    )
                return reconciliation

            # 4. a record: what its interrupted write explains is repaired, from its controller
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

            # 5. every shard holds what the controller holds: the record is cleared, last
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

    def _replicated_table_specs(self, built) -> List[dict]:
        """
        What is compared, table by table: every replicated class that has a table, in the order
        the constructor was given them, then the tag table of every replicated class (a table
        that is neither replicated nor sharded, whose foreign keys other than store_tag's all
        name replicated tables: BackgroundModel_tags).

        Each spec gives the table, the key its rows are compared under and the columns compared.
        A table is keyed by its primary key, and every column but ``timestamp`` is compared
        (every store written before prompt 01 stamps it per shard). BackgroundModelValue is keyed
        by (model_serial, z_serial) and its ``serial`` is not compared, because a store written
        before prompt 01 drew it per shard.
        """
        replicated = [n for n in self._replicated_tables if n in built.tables]
        listed = set(self._replicated_tables) | set(self._sharded_tables.keys())
        tag_tables = []
        for name, table in built.tables.items():
            if name in listed:
                continue
            parents = {fk.column.table.name for fk in table.foreign_keys} - {
                "store_tag"
            }
            if len(parents) > 0 and parents <= set(replicated):
                tag_tables.append(name)

        specs = []
        for name in replicated + tag_tables:
            table = built.tables[name]
            if name == "BackgroundModelValue":
                key = ("model_serial", "z_serial")
                excluded = {"serial", "timestamp"}
            else:
                key = tuple(c.name for c in table.primary_key.columns)
                excluded = {"timestamp"}
            compared = [c.name for c in table.columns if c.name not in excluded]
            specs.append(
                {
                    "name": name,
                    "table": table,
                    "key": key,
                    "compared": compared,
                    "serial_keyed": key == ("serial",),
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
        The primary's replication_in_flight row, as a dict, or None. A store written before
        prompt 01 has no such table, which is read as no record. More than one row is not
        something the write path leaves, and refuses.
        """
        with self._engine.connect() as conn:
            if not sqla.inspect(conn).has_table("replication_in_flight"):
                return None
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
        One shard's replicated rows, table by table: the compared columns the shard's table has,
        and its rows keyed as the spec says, each a tuple in the order of those columns. A table
        the shard lacks is read as empty (an actor would create it empty).
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

            columns = tuple(c for c in spec["compared"] if c in present)
            if not all(k in columns for k in spec["key"]):
                out[name] = {
                    "columns": columns,
                    "rows": {},
                    "duplicates": set(),
                    "unkeyed": True,
                }
                continue

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
        ``columns`` (the shards' tables have different compared columns, or no key column),
        ``duplicate`` (a key held by more than one row on a shard), ``missing`` (a key some
        shards hold and others lack), ``differs`` (a key whose compared values differ between
        the shards that hold it) and ``shard_key`` (a key_serial of the primary's shard_keys that
        is not a wavenumber serial on every shard).
        """
        shard_ids = sorted(snapshots.keys())
        differences = []
        for spec in specs:
            name = spec["name"]
            tables = {sid: snapshots[sid][name] for sid in shard_ids}

            present = {
                sid: t["columns"]
                for sid, t in tables.items()
                if t["columns"] is not None
            }
            if len(set(present.values())) > 1 or any(
                t.get("unkeyed", False) for t in tables.values()
            ):
                differences.append(
                    {"kind": "columns", "table": name, "columns": present}
                )
                continue

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

        # the primary's shard keys name wavenumbers every shard holds
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
        missing under different serials whose other compared values are the same are named as
        a serial split; ``context``, when given, is the whole list of differences those pairs are
        looked for in.
        """
        by_name = {spec["name"]: spec for spec in specs}

        # a serial split: the same row, but for its serial, held under different serials by
        # different shards
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
                if c != "serial"
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
            if kind == "columns":
                entry = {
                    "class_name": d["table"],
                    "shard": sorted(d["columns"].keys()),
                    "key": "its columns",
                    "detail": "the shards' tables do not have the same compared columns, "
                    "or lack their key: "
                    + "; ".join(
                        f"shard {sid}: {list(cols)}"
                        for sid, cols in sorted(d["columns"].items())
                    ),
                }
            elif kind == "duplicate":
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
                    "detail": f"the primary assigns wavenumber serial {d['key']} to shard "
                    f"{d['shard_id']}, but it is missing on shard(s) {d['lackers']}",
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
        class (a BackgroundModel's row, tags and values), only rows the controller holds and
        another shard lacks, written by the recorded write (their timestamp is the record's
        start; under a store_id, that serial), a BackgroundModel as a whole unit, a source or
        response flag the controller holds set after a get, and a validated flag after a
        validate of the record's model, which is recomputed. Anything else is refused.
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

        unit = [cls_name]
        if cls_name == "BackgroundModel":
            unit = [
                t
                for t in (
                    "BackgroundModel",
                    "BackgroundModel_tags",
                    "BackgroundModelValue",
                )
                if t in by_name
            ]
        flag_columns = {
            "redshift": ("source", "response"),
            "wavenumber": ("source", "response"),
        }

        copies = {}  # (table, shard) -> set of keys
        flags = {}  # (table, shard) -> {key: [columns]}
        recompute = False

        for d in differences:
            kind = d["kind"]
            name = d["table"]
            if kind == "shard_key":
                # compared again once the repair is written: a copied wavenumber can satisfy it
                continue
            if kind in ("columns", "duplicate"):
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
                    name == "BackgroundModel"
                    and differing == ["validated"]
                    and turned_on
                    and operation == "validate"
                    and d["key"] == (store_id,)
                ):
                    recompute = True
                else:
                    refuse(
                        {**d, "values": {controller: reference, sid: values}},
                        f"only a source or response flag after a get, or validated after a "
                        f"validate, can be turned on from the controlling shard {controller}; "
                        f"shard {sid} differs in {differing}",
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

        # a BackgroundModel is copied as a unit: a shard lacking the model row receives every
        # one of its tag and value rows, and a shard holding any part of it receives nothing
        if cls_name == "BackgroundModel":
            for sid in sorted(snapshots.keys()):
                models = {k[0] for k in copies.get(("BackgroundModel", sid), set())}
                for name in unit:
                    if name == "BackgroundModel":
                        continue
                    got = copies.get((name, sid), set())
                    held = {
                        k for k in snapshots[controller][name]["rows"] if k[0] in models
                    }
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
                            "BackgroundModel it lacks",
                        )
                    for key in sorted(held - got):
                        refuse(
                            {
                                "kind": "missing",
                                "table": "BackgroundModel",
                                "key": (key[0],),
                                "holders": [controller],
                                "lackers": [sid],
                            },
                            f"shard {sid} already holds part of this model ({name} {key}), "
                            "and a BackgroundModel is copied only as a unit",
                        )

            # value serials are the controller's, and must be free on the shard
            for (name, sid), keys in sorted(copies.items()):
                if name != "BackgroundModelValue":
                    continue
                table = by_name[name]["table"]
                serials = sorted(rows[name][k]["serial"] for k in keys)
                taken = set(
                    conns[sid]
                    .execute(
                        sqla.select(table.c.serial).filter(
                            table.c.serial.between(serials[0], serials[-1])
                        )
                    )
                    .scalars()
                    .all()
                ) & set(serials)
                conns[sid].rollback()
                if len(taken) > 0:
                    refusals.append(
                        {
                            "class_name": name,
                            "shard": sid,
                            "key": f"serials {sorted(taken)}",
                            "detail": "these value serials of the controller's model are "
                            "already taken on the shard",
                        }
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
        """The controller's rows under ``keys``, every column its table has, as mappings."""
        table = spec["table"]
        present = {
            row[1]
            for row in conn.exec_driver_sql(f'PRAGMA table_info("{spec["name"]}")')
        }
        columns = [c for c in table.columns if c.name in present]
        first = spec["key"][0]
        out = {}
        for row in conn.execute(
            sqla.select(*columns).filter(
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
        so that a model row precedes its tags and values), then the flags, then the recomputed
        validated flags. Return (actions, refusals).
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
        After a validate of the record's BackgroundModel, recompute its ``validated`` flag on each
        shard by the factory's own rule (sqla_BackgroundModelFactory.validate: its value-row count
        against z_samples), from that shard's own value rows, each inside a savepoint. A result
        that turns the flag on is kept; any other is rolled back to the savepoint, so that no flag
        is written False. A shard holding True whose rule gives False refuses. Return (actions,
        refusals); whether the shards then agree is for the second comparison.
        """
        factory = _factories["BackgroundModel"]
        table = built.tables["BackgroundModel"]
        serial = record["store_id"]

        class _StoredModel(DatastoreObject):
            """What the factory's validate reads of a model: its serial and its label."""

            def __init__(self, store_id: int, label):
                super().__init__(store_id)
                self.label = label

        actions = []
        refusals = []
        for sid in sorted(conns.keys()):
            snap = snapshots[sid]["BackgroundModel"]
            row = snap["rows"].get((serial,))
            if row is None:
                continue
            columns = snap["columns"]
            stored = row[columns.index("validated")] is True
            label = row[columns.index("label")] if "label" in columns else None

            savepoint = conns[sid].begin_nested()
            result = factory.validate(
                _StoredModel(serial, label),
                conn=conns[sid],
                table=table,
                tables=built.tables,
            )
            if result is True and not stored:
                savepoint.commit()
                actions.append(
                    {
                        "action": "validated recomputed",
                        "class_name": "BackgroundModel",
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
                            "class_name": "BackgroundModel",
                            "shard": sid,
                            "key": f"serial {serial}",
                            "detail": "the shard holds validated=True, and the factory's rule "
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

        Every record goes through the one resolver (Datastore/shard_paths.py). A legacy absolute
        record is read as the sibling of that name; the absolute path itself is never used. The
        rows are not rewritten.
        """
        relocated_legacy = []
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

            if is_legacy_record(stored) and Path(stored) != filename:
                relocated_legacy.append(stored)

        if len(relocated_legacy) > 0:
            legacy_dirs = ", ".join(
                f'"{d}"'
                for d in sorted({str(Path(s).parent) for s in relocated_legacy})
            )
            print(
                f'!! Primary database "{str(primary_file)}" records {len(relocated_legacy)} shard(s) by legacy absolute path in {legacy_dirs}; reading them as siblings in "{str(primary_file.parent)}" instead (stored records not rewritten)'
            )

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
        `shards.filename` of the destination primary is changed. A legacy source's absolute rows
        become bare names at the destination, because that is the operation asked for.
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
        legacy absolute record is read as the sibling of that name in the primary's directory, and
        the absolute path itself is never opened, so a primary whose records name another store's
        files deletes its own siblings and never those.

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
        primary: Path, verb: str, *, missing_ok: bool = False
    ) -> Tuple[Dict[int, Path], Dict[int, str]]:
        """
        Read the `shards` rows of the closed store ``primary`` (opened mode=ro) and check the files
        they name, through the same two methods the constructor uses. Return (serial -> path,
        serial -> stored record), or raise RuntimeError naming the store and the problem.
        """
        try:
            conn = sqlite3.connect(f"{primary.as_uri()}?mode=ro", uri=True)
            try:
                rows = conn.execute("SELECT serial, filename FROM shards").fetchall()
            finally:
                conn.close()
        except sqlite3.Error as e:
            raise RuntimeError(
                f'Cannot {verb} sharded datastore "{str(primary)}": its shards table could not be read ({e})'
            ) from e

        files: Dict[int, Path] = {}
        records: Dict[int, str] = {}
        try:
            ShardedPool._resolve_shard_rows(primary, rows, files, records)
        except RuntimeError as e:
            raise RuntimeError(
                f'Cannot {verb} sharded datastore "{str(primary)}": {e}'
            ) from e

        # missing_ok is for delete_store(resume=True) only: a shard file that is not there is not
        # refused. Every serial is still returned, present or not
        problems = ShardedPool._shard_file_problems(
            files, records, missing_ok=missing_ok
        )
        if len(problems) > 0:
            raise RuntimeError(
                f'Cannot {verb} sharded datastore "{str(primary)}": '
                + "; ".join(problems)
            )

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
            src_files, src_records = ShardedPool._read_closed_store(src_primary, mode)
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
        files, records = ShardedPool._read_closed_store(primary, plan.mode)
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
                primary_file, "delete", missing_ok=resume
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
    # the get, store and validate of a replicated class take, and the one place that writes and
    # clears the primary's replication_in_flight record (prompts/datastore-integrity, prompt 01):
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

    @staticmethod
    def _value_serials(obj) -> Optional[List[Optional[int]]]:
        """The serials of an object's value rows, if it carries any (BackgroundModel), else None."""
        values = getattr(obj, "_values", None)
        if values is None:
            return None
        return [getattr(value, "_my_id", None) for value in values]

    def _store_impl_replicated_table(self, cls_name, item):
        # we push an initial 'store' to the controlling shard, and then push the object, complete
        # with its new store_id (and its value rows' serials), to all the other shards, each of
        # which inserts it under those serials or verifies that it already holds it
        def submit_controller(handle, timestamp):
            return handle.object_store.remote(item, insert_timestamp=timestamp)

        def replication(controller, obj):
            if not hasattr(obj, "_my_id") or obj._my_id is None:
                raise RuntimeError(
                    f'Stored object of type "{cls_name}" was not assigned a store_id field'
                )

            serial = obj._my_id
            value_serials = ShardedPool._value_serials(obj)

            def submit(handle, timestamp):
                return handle.object_store.remote(obj, insert_timestamp=timestamp)

            def check(shard_id, answer):
                got = getattr(answer, "_my_id", None)
                if got != serial:
                    raise ReplicationMismatch(shard_id, cls_name, serial, got)
                got_values = ShardedPool._value_serials(answer)
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
