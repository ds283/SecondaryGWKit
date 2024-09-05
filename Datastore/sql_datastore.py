from datetime import datetime
from pathlib import Path
from os import PathLike
from typing import Optional, Union, Iterable

from ray import remote
from ray.data import Dataset

import sqlalchemy as sqla


VERSION_ID_LENGTH = 64


PathType = Union[str, PathLike]


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

        # initialize empty dict of registered storable classes
        self._registered_classes = {}

        # initialize empty dict of storage schema
        # each record collects SQLAlchemy column and table definitions, queries, etc., for a registered storable class
        self._schema = {}

    def _create_engine(self, db_name: PathType, expect_exists: bool = False):
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

        self._engine = sqla.create_engine(f"sqlite:///{db_name}", future=True)
        self._metadata = sqla.MetaData()

        self._version_table = sqla.Table(
            "versions",
            self._metadata,
            sqla.Column("serial", sqla.Integer, primary_key=True, nullable=False),
            sqla.Column("version_id", sqla.String(VERSION_ID_LENGTH)),
        )

        self._register_schema()

    def create_datastore(self, db_name: PathType):
        """
        Create and initialize an empty data container. Assumes the container not to be physically present at the specified path
        and will fail with an error if it is
        :param db_name: path to on-disk database
        :return:
        """
        self._ensure_no_engine()
        self._create_engine(db_name, expect_exists=False)

        print("-- creating database tables")

        # generate internal tables
        self._version_table.create(self._engine)
        self._version_serial = self._make_version_serial()

        # generate tables defined by any registered storable classes
        self._create_storage_tables()

    def _make_version_serial(self):
        if self._engine is None:
            raise RuntimeError(
                "No database connection exists in _make_version_serial()"
            )

        with self._engine.begin() as conn:
            result = conn.execute(
                sqla.select(self._version_table.c.serial).filter(
                    self._version_table.c.version_id == self._version_id
                )
            )

            x = result.scalar()

            if x is not None:
                return x

            serial = conn.execute(
                sqla.select(sqla.func.count()).select_from(self._version_table)
            ).scalar()
            conn.execute(
                sqla.insert(self._version_table),
                {"serial": serial, "version_id": self._version_id},
            )

            conn.commit()

        return serial

    def _ensure_engine(self):
        if self._engine is None:
            raise RuntimeError(f"No storage engine has been initialized")

    def _ensure_no_engine(self):
        if self._engine is not None:
            raise RuntimeError(f"A storage engine has already been initialized")

    def open_datastore(self, db_name: str) -> None:
        self._ensure_no_engine()
        self._create_engine(db_name, expect_exists=True)

        self._version_serial = self._make_version_serial()

    def register_storable_classes(self, ClassObjectList):
        if not isinstance(ClassObjectList, Iterable):
            ClassObjectList = {ClassObjectList}

        for ClassObject in ClassObjectList:
            class_name = ClassObject.__name__

            if class_name in self._registered_classes:
                raise RuntimeWarning(
                    f"Duplicate attempt to register storable class '{class_name}'"
                )

            self._registered_classes[class_name] = ClassObject
            print(f"Registered storable class '{class_name}'")

    def _register_schema(self):
        self._ensure_engine()

        # iterate through all registered classes, querying them for the columns
        # they need to store their data
        for cls_name, cls in self._registered_classes.items():
            if cls_name in self._schema:
                raise RuntimeWarning(
                    f"Duplicate attempt to register table for storable class '{cls_name}'"
                )

            schema = {}

            # query class for a list of columns that it wants to store
            class_data = cls.generate_columns()

            defer_insert = class_data.get("defer_insert", False)
            schema["defer_insert"] = defer_insert

            # generate basic table
            # metadata can be controlled by the owning class

            serial_col = sqla.Column("serial", sqla.Integer, primary_key=True)
            tab = sqla.Table(
                cls_name,
                self._metadata,
                serial_col,
            )

            # attach pre-defined columns
            use_version = class_data.get("version", False)
            schema["use_version"] = use_version
            if use_version:
                version_col = sqla.Column(
                    "version", sqla.Integer, sqla.ForeignKey("versions.serial")
                )
                tab.append_column(version_col)
                schema["version_col"] = version_col

            use_timestamp = class_data.get("timestamp", False)
            schema["use_timestamp"] = use_timestamp
            if use_timestamp:
                timestamp_col = sqla.Column("timestamp", sqla.DateTime())
                tab.append_column(timestamp_col)
                schema["timestamp_col"] = timestamp_col

            use_stepping = class_data.get("stepping", False)
            if isinstance(use_stepping, str):
                if use_stepping not in ["minimum", "exact"]:
                    print(f"Warning: ignored stepping selection '{use_stepping}' when registering storable class '{cls_name}'")
                    use_stepping = False

            _use_stepping = isinstance(use_stepping, str) or use_stepping is not False
            _stepping_mode = None if not isinstance(use_stepping, str) else use_stepping
            schema["use_stepping"] = _use_stepping
            schema["stepping_mode"] = _stepping_mode

            if _use_stepping:
                stepping_col = sqla.Column("stepping", sqla.Integer)
                tab.append_column(stepping_col)
                schema["stepping_col"] = stepping_col

            # append all columns supplied by the class
            sqla_columns = class_data.get("columns", [])
            for col in sqla_columns:
                tab.append_column(col)

            # store in table cache
            schema["serial_col"] = serial_col
            schema["columns"] = sqla_columns
            schema["table"] = tab

            # generate and cache lookup query
            full_columns = [serial_col]
            if use_version:
                full_columns.append(version_col)
            if use_timestamp:
                full_columns.append(timestamp_col)
            if _use_stepping:
                full_columns.append(stepping_col)
            full_columns.extend(sqla_columns)

            full_query = sqla.select(*full_columns).select_from(tab)
            schema["full_query"] = full_query

            serial_query = sqla.select(serial_col).select_from(tab)
            schema["serial_query"] = serial_query

            self._schema[cls_name] = schema
            print(f"Registered storage schema from storable class '{cls_name}'")

    def _create_storage_tables(self):
        self._ensure_engine()

        for record in self._schema.values():
            tab = record["table"]
            tab.create(self._engine)

    def _ensure_registered_schema(self, cls_name: str):
        if cls_name not in self._registered_classes:
            raise RuntimeError(
                f'No storable class of type "{cls_name}" has been registered'
            )

    def query(self, item, serial_only: bool = True):
        self._ensure_engine()

        cls_name = type(item).__name__
        self._ensure_registered_schema(cls_name)

        record = self._schema[cls_name]
        tab = record["table"]

        uses_timestamp = record.get("use_timestamp", False)
        uses_version = record.get("use_version", False)
        uses_stepping = record.get("use_stepping", False)
        defer_insert = record.get("defer_insert", False)

        if serial_only:
            query = record["serial_query"]
        else:
            query = record["full_query"]

        # pass basic query back to item instance to add .filter() specifications, or any other adjustments that are needed
        query = item.build_query(tab, query)

        # filter by supplied stepping, if this class uses that metadata
        if uses_stepping:
            stepping_mode = record["stepping_mode"]
            if stepping_mode == "exact":
                query = query.filter(tab.c.stepping == item.stepping)
            elif stepping_mode == "minimum":
                query = query.filter(tab.c.stepping >= item.stepping)


        with self._engine.begin() as conn:
            # print(f"QUERY: {query}")
            # print(f"QUERY COLUMNS {query.columns.keys()}")
            result = conn.execute(query)

            if serial_only:
                x = result.scalar()
                if x is not None:
                    return x
            else:
                x = result.one_or_none()
                if x is not None:
                    # store _asdict() rather than SQLAlchemy Row object in an attempt to produce simpler payloads for Ray
                    return {"store_id": x.serial, "data": x._asdict()}

            if defer_insert:
                return None

            data = item.build_storage_payload()

            if uses_version:
                data = data | {"version": self._version_id}
            if uses_timestamp:
                data = data | {"timestamp": datetime.now()}
            if uses_stepping:
                data = data | {"stepping": item.stepping}

            obj = conn.execute(
                sqla.dialects.sqlite.insert(tab).on_conflict_do_nothing(), data
            )
            conn.commit()

            serial = obj.lastrowid
            if serial is not None:
                if serial_only:
                    return serial
                else:
                    data = data | {"serial": serial}
                    return {"store_id": serial, "data": data}

        raise RuntimeError(
            f"Insert error when querying for storable class '{cls_name}'"
        )

    def table(self, ClassObject) -> sqla.Table:
        """
        Obtain the SQLAlchemy table object corresponding to a datastore object
        """
        cls_name: str = ClassObject.__name__
        self._ensure_registered_schema(cls_name)

        record = self._schema[cls_name]
        tab = record["table"]

        return tab
