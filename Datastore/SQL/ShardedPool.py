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
from Datastore.SQL.Datastore import PathType, ReadTableConfigType
from Datastore.SQL.ProfileAgent import ProfileAgent
from Datastore.SQL.SerialPoolBroker import SerialPoolBroker
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
            self._create_engine()
            self._read_shard_data()

            # fail closed, before any actor exists: a Datastore actor given a missing shard file
            # creates an empty database there, and the pool would open with nothing in it
            self._check_shard_files()

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

    def _write_shard_data(self):
        self._shard_file_table.create(self._engine)
        self._shard_key_config_table.create(self._engine)
        self._shard_key_table.create(self._engine)
        self._replicated_tables_table.create(self._engine)
        self._sharded_tables_table.create(self._engine)

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
        shard_db_files: Dict[int, PathType], shard_records: Dict[int, str]
    ) -> List[str]:
        """
        Return one message for each resolved shard that is not a usable file (missing, not a
        regular file, a symbolic link: Datastore/shard_paths.py shard_file_problem), and for each
        pair of serials that resolve to the same file. An empty list means every shard is usable.
        """
        problems = []
        for serial, path in sorted(shard_db_files.items()):
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

    # COPY OR MOVE A CLOSED STORE
    #
    # A store is its primary and its shards, and nothing else: these methods copy, move, check and
    # mention no other file. They are static and work on a closed store. An open pool has one
    # Datastore actor per shard holding its file, so they are not methods of an open pool, and they
    # start no Ray, create no actor and need no instance. They cannot tell whether some process
    # has the store open (a rollback-journal store leaves no file while idle); making sure nothing
    # is using it is the caller's job.
    #
    # They never delete a file, never overwrite one, and never write the source. On failure they
    # do not clean up: they raise, naming the step that failed and the store files that exist at
    # each end. Deleting is for a person.

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
    def _journal_paths(path: Path) -> List[Path]:
        """The names SQLite gives the rollback journal and the WAL files of the database ``path``."""
        return [
            path.with_name(path.name + suffix) for suffix in _SQLITE_JOURNAL_SUFFIXES
        ]

    @staticmethod
    def _read_closed_store(
        primary: Path, verb: str
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

        problems = ShardedPool._shard_file_problems(files, records)
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

    def _get_impl_replicated_table(self, cls_name, kwargs):
        # pick a shard id at random to be the "controlling" shard.
        # we will push an initial 'get' to this controlling shard.
        # if a new database object was created by the get, we then have to push a replica
        # to all the other shards
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

        # was this a vectorized get?
        if "payload_data" in kwargs:
            payload_data = kwargs["payload_data"]

            if len(payload_data) != len(objects):
                raise RuntimeError(
                    f"object_get() data returned from selected datastore (shared={shard_key}) has a different length (length={len(objects)}) to payload data (length={len(payload_data)})"
                )

            # add explicit serial specifier
            new_payload = []
            for i in range(len(payload_data)):
                # if this object has a valid store_id and has the _new_insert or _updated metadata
                # flags set, push to all the remaining shards in order to keep them in sync
                if (
                    hasattr(objects[i], "_my_id")
                    and objects[i]._my_id is not None
                    and (
                        hasattr(objects[i], "_new_insert")
                        or hasattr(objects[i], "_updated")
                    )
                ):
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

            # if this object has a valid store_id and has the _new_insert or _updated
            # metadata flags set, push to all remaining shards in order to keep them in sync
            if (
                hasattr(objects, "_my_id")
                and objects._my_id is not None
                and (hasattr(objects, "_new_insert") or hasattr(objects, "_updated"))
            ):
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
        if cls_name == self._ShardKeyType_name:
            self._assign_shard_keys(objects)

        # return original object (we just discard any copies returned from other shards)
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

    def _store_impl_replicated_table(self, cls_name, item):
        # pick a shard id at random to be the "controlling" shard
        # we will push an initial 'store' to this controlling shard.
        # if a new database object was created by the get, we then have to push a replica
        # to all the other shards
        shard_ids = list(self._shards.keys())
        i = random.randrange(len(shard_ids))

        # swap this entry with the last element, then pop it
        shard_ids[i], shard_ids[-1] = shard_ids[-1], shard_ids[i]
        shard_key = shard_ids.pop()

        ref = self._shards[shard_key].object_store.remote(item)
        obj = ray.get(ref)

        # now push the object, complete with its new 'store_id', to all the other shards
        if not hasattr(obj, "_my_id") or obj._my_id is None:
            raise RuntimeError(
                f'Stored object of type "{cls_name}" was not assigned a store_id field'
            )

        ray.get([self._shards[key].object_store.remote(obj) for key in shard_ids])

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
        # pick a shard id at random to be the "controlling" shard
        # we will push an initial 'validate' to this controlling shard.
        shard_ids = list(self._shards.keys())
        i = random.randrange(len(shard_ids))

        # swap this entry with the last element, then pop it
        shard_ids[i], shard_ids[-1] = shard_ids[-1], shard_ids[i]
        shard_key = shard_ids.pop()

        ref = self._shards[shard_key].object_validate.remote(item)
        outcome = ray.get(ref)

        # if object did not validate, do not push validation requests to remaining shards
        if outcome is False or outcome is None:
            return [ref]

        outcomes = ray.get(
            [self._shards[key].object_validate.remote(item) for key in shard_ids]
        )
        if any(oc is not True for oc in outcomes):
            print(f"!! Validation outcomes did not agree between shards:")
            print(f"|    outcomes = {outcomes}")
            raise RuntimeError(
                f'Object validation produced different outcomes on different shards for replicated object of type "{cls_name}"'
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
