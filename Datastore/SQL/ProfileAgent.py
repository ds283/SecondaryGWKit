import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import ray
import sqlalchemy as sqla
from sqlalchemy.exc import SQLAlchemyError

from defaults import DEFAULT_STRING_LENGTH


@ray.remote
class ProfileAgent:
    def __init__(
        self, db_name: str, timeout: Optional[int] = None, label: Optional[str] = None
    ):
        self._timeout = timeout

        self._db_file = Path(db_name).resolve()

        if self._db_file.is_dir():
            raise RuntimeError(
                f'Specified profile agent database file "{str(self._db_file)}" is a directory"'
            )
        elif not self._db_file.exists():
            # create parent directories if they do not already exist
            self._db_file.parents[0].mkdir(parents=True, exist_ok=True)
            self._create_engine()
            self._ensure_tables()
        else:
            self._create_engine()
            self._ensure_tables()

        if label is not None:
            self._label = label
        else:
            self._label = (
                f"job-start-{datetime.now().replace(microsecond=0).isoformat()}"
            )
        with self._engine.begin() as conn:
            obj = conn.execute(sqla.insert(self._job_table), {"label": self._label})

            self._job_id = obj.lastrowid

    def _create_engine(self):
        connect_args = {}
        if self._timeout is not None:
            connect_args["timeout"] = self._timeout

        self._engine = sqla.create_engine(
            f"sqlite:///{str(self._db_file)}",
            future=True,
            connect_args=connect_args,
        )
        self._metadata = sqla.MetaData()
        self._inspector = sqla.inspect(self._engine)

        self._job_table = sqla.Table(
            "jobs",
            self._metadata,
            sqla.Column("serial", sqla.Integer, primary_key=True),
            sqla.Column("label", sqla.String(DEFAULT_STRING_LENGTH)),
        )
        self._profile_table = sqla.Table(
            "profile_data",
            self._metadata,
            sqla.Column("serial", sqla.Integer, primary_key=True),
            sqla.Column(
                "job_serial",
                sqla.Integer,
                sqla.ForeignKey("jobs.serial"),
                index=True,
                nullable=False,
            ),
            sqla.Column(
                "source",
                sqla.String(DEFAULT_STRING_LENGTH),
                nullable=False,
            ),
            sqla.Column("method", sqla.String(DEFAULT_STRING_LENGTH), nullable=False),
            sqla.Column("start_time", sqla.DateTime(), nullable=False),
            sqla.Column("elapsed", sqla.Float(64), nullable=False),
            sqla.Column("metadata", sqla.String(DEFAULT_STRING_LENGTH)),
        )

    def _ensure_tables(self):
        if not self._inspector.has_table("jobs"):
            self._job_table.create(self._engine)

        if not self._inspector.has_table("profile_data"):
            self._profile_table.create(self._engine)

    def write_batch(self, batch):
        try:
            with self._engine.begin() as conn:
                for item in batch:
                    obj = conn.execute(
                        sqla.insert(self._profile_table),
                        {
                            "job_serial": self._job_id,
                            "source": item["source"],
                            "method": item["method"],
                            "start_time": item["start_time"],
                            "elapsed": item["elapsed"],
                            "metadata": item["metadata"],
                        },
                    )
        except SQLAlchemyError as e:
            print(f"!! ProfileAgent: insert error, payload = {batch}")
            raise e

    def clean_up(self) -> None:
        if self._engine is not None:
            self._engine.dispose()


class ProfileBatcher:
    def __init__(
        self, profile_agent, source_label: str, batch_size: Optional[int] = 100
    ):
        self._profile_agent = profile_agent
        self._source_label = source_label

        self._batch_size = batch_size
        self._batch = []

    def write(self, method: str, start_time: datetime, elapsed: float, metadata: str):
        if self._profile_agent is None:
            return

        self._batch.append(
            {
                "source": self._source_label,
                "method": method,
                "start_time": start_time,
                "elapsed": elapsed,
                "metadata": metadata,
            }
        )

        if len(self._batch) > self._batch_size:
            self._push_batch()

    def _push_batch(self):
        if len(self._batch) > 0:
            self._profile_agent.write_batch.remote(self._batch)
            self._batch = []

    def clean_up(self) -> None:
        self._push_batch()


class ProfileBatchManager:
    def __init__(
        self,
        batcher,
        method: str,
        metadata: Optional[dict] = {},
        num_items: Optional[int] = None,
    ):
        self._batcher = batcher

        if method is None:
            raise RuntimeError("method cannot be None")

        if not isinstance(metadata, dict):
            raise RuntimeError("metadata should be a dict")

        self._method = method
        self._metadata = metadata

        self._start_time = None
        self._perf_timer_start = None

        self._num_items = num_items

    def __enter__(self):
        self.start_time = datetime.now()
        self.perf_timer_start = time.perf_counter()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.perf_timer_start

        if self._num_items is not None:
            self._metadata.update(
                {
                    "num_items": self._num_items,
                    "time_per_item": elapsed / self._num_items,
                }
            )

        self._batcher.write(
            method=self._method,
            start_time=self.start_time,
            elapsed=elapsed,
            metadata=str(self._metadata) if len(self._metadata) > 0 else None,
        )

    def update_metadata(self, data):
        self._metadata.update(data)

    def update_num_items(self, num_items: int):
        self._num_items = num_items
