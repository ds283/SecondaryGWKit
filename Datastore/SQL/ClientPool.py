import time
from typing import Optional

import ray
from ray.actor import ActorHandle

from Datastore.SQL.ProfileAgent import ProfileBatcher, ProfileBatchManager

_default_serial_batch_size = {
    "store_tag": 5,
    "redshift": 200,
    "wavenumber": 100,
    "wavenumber_exit_time": 100,
    "tolerance": 5,
    "LambdaCDM": 5,
    "IntegrationSolver": 6,
    "BackgroundModel": 5,
    "BackgroundModelValue": 100,
    "TkNumericalIntegration": 50,
    "TkNumerical_tags": 10,
    "TkNumericalValue": 5000,
    "GkNumericalIntegration": 50,
    "GkNumerical_tags": 10,
    "GkNumericalValue": 5000,
    "QuadSource": 50,
    "QuadSource_tags": 10,
    "QuadSourceValue": 5000,
    "GkWKBIntegration": 50,
    "GkWKB_tags": 10,
    "GkWKBValue": 5000,
    "GkSource": 50,
    "GkSource_tags": 10,
    "GkSourceValue": 5000,
    "QuadSourceIntegral": 50,
    "QuadSourceIntegral_tags": 10,
    "QuadSourceIntegralValue": 5000,
    "OneLoopIntegral": 50,
    "OneLoopIntegral_tags": 10,
    "OneLoopIntegralValue": 5000,
}


class ClientPool:
    def __init__(
        self,
        table: str,
        profiler: ProfileBatcher,
        broker: Optional[ActorHandle] = None,
        default_batch_size: int = 500,
    ):
        self._table = table
        self._broker = broker
        self._profiler = profiler
        self._batch_size = default_batch_size

        self._pool = set()
        self._leased = set()
        self._committed = set()

    def lease_serial(self) -> int:
        if len(self._pool) == 0:
            if len(self._committed) > 0:
                with ProfileBatchManager(
                    self._profiler,
                    "ClientPool.lease_serial|commit_serials",
                    {"table": self._table},
                ) as mgr:
                    ray.get(
                        self._broker.commit_serials.remote(self._table, self._committed)
                    )
                    self._committed = set()

            with ProfileBatchManager(
                self._profiler,
                "ClientPool.lease_serial|lease_serials",
                {"table": self._table},
            ) as mgr:
                self._pool = ray.get(
                    self._broker.lease_serials.remote(self._table, self._batch_size)
                )

        leased_serial = self._pool.pop()
        if leased_serial is None:
            raise RuntimeError("ClientPool: leased serial number is None")

        self._leased.add(leased_serial)
        return leased_serial

    def release_serial(self, serial: int) -> bool:
        if serial not in self._leased:
            print(
                f'!! ClientPool (table="{self._table}"): attempt to release serial #{serial} for table "{self._table}", but this has not been leased'
            )
            return False

        self._leased.remove(serial)
        self._pool.add(serial)
        return True

    def commit_serial(self, serial: int) -> bool:
        if serial not in self._leased:
            print(
                f'!! ClientPool (table="{self._table}"): attempt to commit serial #{serial} for table "{self._table}", but this has not been leased'
            )
            return False

        self._leased.remove(serial)
        self._committed.add(serial)
        return True

    def release(self) -> None:
        with ProfileBatchManager(
            self._profiler, "ClientPool.release", {"table": self._table}
        ) as mgr:
            ray.get(
                [
                    self._broker.commit_serials.remote(self._table, self._committed),
                    self._broker.release_serials.remote(self._table, self._pool),
                ]
            )

        self._pool = set()
        self._committed = set()

    def clean_up(self) -> None:
        self.release()


# default interval at which to sync leased/committed serial numbers with the central broker
# currently 3 minutes
DEFAULT_SERIAL_SYNC_INTERVAL = 180


class SerialPoolManager:
    def __init__(
        self,
        profiler: ProfileBatcher,
        broker: Optional[ActorHandle] = None,
        default_sync_interval: int = DEFAULT_SERIAL_SYNC_INTERVAL,
    ):
        self._broker = broker
        self._profiler = profiler

        self._tables = {}

        self._sync_interval = default_sync_interval
        self._last_sync = time.time()

    def lease_serial(self, table: str) -> Optional[int]:
        if self._broker is None:
            return None

        if table not in self._tables:
            self._tables[table] = ClientPool(
                table,
                self._profiler,
                self._broker,
                default_batch_size=_default_serial_batch_size[table],
            )

        serial = self._tables[table].lease_serial()
        self._check_sync()
        return serial

    def release_serial(self, table: str, serial: Optional[int]) -> bool:
        if self._broker is None:
            return True

        if table not in self._tables:
            raise RuntimeError(
                f'SerialPoolManager: release_serial() called on non-existent table "{table}"'
            )

        outcome = self._tables[table].release_serial(serial)
        self._check_sync()
        return outcome

    def commit_serial(self, table: str, serial: Optional[int]) -> bool:
        if self._broker is None:
            return True

        if table not in self._tables:
            raise RuntimeError(
                f'SerialPoolManager: commit_serial() called on non-existent table "{table}"'
            )

        outcome = self._tables[table].commit_serial(serial)
        self._check_sync()
        return outcome

    def _check_sync(self) -> None:
        current_time = time.time()
        if current_time - self._last_sync < self._sync_interval:
            return

        for pool in self._tables.values():
            pool.release()

        self._last_sync = current_time

    def clean_up(self) -> None:
        for pool in self._tables.values():
            pool.clean_up()


class SerialLeaseManager:
    def __init__(self, manager: SerialPoolManager, table: str, obtain_lease: bool):
        self._manager = manager
        self._table = table

        self.serial: Optional[int] = None

        self._release_on_exit: bool = False
        self._obtain_lease: bool = obtain_lease

    def __enter__(self):
        if self._obtain_lease:
            self.serial = self._manager.lease_serial(self._table)

            if self.serial is None:
                raise RuntimeError(
                    f'SerialLeaseManager (table="{self._table}"): leased serial number is None'
                )
            self._release_on_exit = True

        return self

    def commit(self):
        if self._release_on_exit is False:
            print(
                "!! SerialLeaseManage: API error, commit() called when no release on exit is required"
            )
            return

        if self.serial is None:
            print(
                "!! SerialLeaseManage: API error, commit() called when no active serial number is being managed"
            )
            self._release_on_exit = False
            return

        self._manager.commit_serial(self._table, self.serial)
        self._release_on_exit = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._release_on_exit:
            self._manager.release_serial(self._table, self.serial)
