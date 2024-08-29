from pathlib import Path
from os import PathLike
from typing import Optional, Union

from ray import remote
from ray.data import Dataset

import sqlalchemy as sqla


VERSION_ID_LENGTH = 64
DEFAULT_STRING_LENGTH = 256


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
            sqla.Column("serial", sqla.Integer, primary_key=True),
            sqla.Column("version_id", sqla.String(VERSION_ID_LENGTH)),
        )

        self._lambda_CDM_model_table = sqla.Table(
            "lambda_cdm_models",
            self._metadata,
            sqla.Column("serial", sqla.Integer, primary_key=True),
            sqla.Column("name", sqla.String(DEFAULT_STRING_LENGTH)),
            sqla.Column("omega_m", sqla.Float(64)),
            sqla.Column("omega_cc", sqla.Float(64)),
            sqla.Column("h", sqla.Float(64)),
            sqla.Column("f_baryon", sqla.Float(64)),
            sqla.Column("T_CMB", sqla.Float(64)),
            sqla.Column("Neff", sqla.Float(64)),
        )

    def create_datastore(self, db_name: PathType):
        """
        Create and initialize an empty data container. Assumes the container not to be physically present at the specified path
        and will fail with an error if it is
        :param db_name: path to on-disk database
        :return:
        """
        if self._engine is not None:
            raise RuntimeError(
                "create_datastore() called when a storage engine already exists"
            )

        self._create_engine(db_name, expect_exists=False)

        print("-- creating database tables")

        self._version_table.create(self._engine)
        self._version_serial = self._make_version_serial()

        self._lambda_CDM_model_table.create(self._engine)

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

    def open_datastore(self, db_name: str) -> None:
        if self._engine is not None:
            raise RuntimeError(
                "open_datastore() called when a database engine already exists"
            )

        self._create_engine(db_name, expect_exists=True)
        self._version_serial = self._make_version_serial()
