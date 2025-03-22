import time
from typing import Optional

from CosmologyConcepts import wavenumber
from utilities import format_time
from .base import (
    IntegrationSupervisor,
    DEFAULT_UPDATE_INTERVAL,
)


class ThetaSupervisor(IntegrationSupervisor):
    def __init__(
        self,
        k: wavenumber,
        z_init: float,
        z_target: float,
        label: str,
        notify_interval: int = DEFAULT_UPDATE_INTERVAL,
    ):
        super().__init__(notify_interval)

        self._label: str = label

        self._k: wavenumber = k
        self._z_init: float = z_init
        self._z_target: float = z_target

        self._z_range: float = self._z_init - self._z_target

        self._last_z: float = self._z_init

        self._WKB_violation: bool = False
        self._WKB_violation_z: Optional[float] = None
        self._WKB_violation_efolds_subh: Optional[float] = None

        self._nfev: int = 0

    def __enter__(self):
        super().__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)

    def message(self, current_z, msg):
        current_time = time.time()
        since_last_notify = current_time - self._last_notify
        since_start = current_time - self._start_time

        update_number = self.report_notify()

        z_complete = self._z_init - current_z
        z_remain = self._z_range - z_complete
        percent_remain = z_remain / self._z_range
        print(
            f"** STATUS UPDATE #{update_number}: {self._label} WKB theta_k(z) integration for k = {self._k.k_inv_Mpc:.5g}/Mpc (store_id={self._k.store_id}) has been running for {format_time(since_start)} ({format_time(since_last_notify)} since last notification)"
        )
        print(
            f"|    current z={current_z:.5g} (initial z={self._z_init:.5g}, target z={self._z_target:.5g}, z complete={z_complete:.5g}, z remain={z_remain:.5g}, {percent_remain:.3%} remains)"
        )
        if self._last_z is not None:
            z_delta = self._last_z - current_z
            print(f"|    redshift advance since last update: Delta z = {z_delta:.5g}")
        print(
            f"|    {self.RHS_evaluations} RHS evaluations, mean {self.mean_RHS_time:.5g}s per evaluation, min RHS time = {self.min_RHS_time:.5g}s, max RHS time = {self.max_RHS_time:.5g}s"
        )
        print(f"|    {msg}")

        self._last_z = current_z

    def report_WKB_violation(self, z: float, efolds_subh: float):
        if self._WKB_violation:
            return

        print(
            f"!! WARNING: {self._label} WKB theta_k(z) integration for k = {self._k.k_inv_Mpc:.5g}/Mpc (store_id={self._k.store_id}) may have violated the validity criterion for the WKB approximation"
        )
        print(f"|    current z={z:.5g}, e-folds inside horizon={efolds_subh:.3g}")
        self._WKB_violation = True
        self._WKB_violation_z = z
        self._WKB_violation_efolds_subh = efolds_subh

    @property
    def has_WKB_violation(self) -> bool:
        return self._WKB_violation

    @property
    def WKB_violation_z(self) -> float:
        return self._WKB_violation_z

    @property
    def WKB_violation_efolds_subh(self) -> float:
        return self._WKB_violation_efolds_subh

    def notify_new_nfev(self, nfev: int):
        self._nfev += nfev

    @property
    def nfev(self) -> int:
        return self._nfev

    @property
    def z_init(self) -> float:
        return self._z_init

    @property
    def z_target(self) -> float:
        return self._z_target


class QSupervisor(IntegrationSupervisor):
    def __init__(
        self,
        k: wavenumber,
        u_init: float,
        u_target: float,
        label: str,
        notify_interval: int = DEFAULT_UPDATE_INTERVAL,
    ):
        super().__init__(notify_interval)

        self._label: str = label

        self._k: wavenumber = k
        self._u_init: float = u_init
        self._u_target: float = u_target

        self._u_range: float = self._u_target - self._u_init

        self._last_u: float = self._u_init

        self._largest_Q: Optional[float] = 0
        self._smallest_Q: Optional[float] = 0

        self._WKB_violation: bool = False
        self._WKB_violation_z: Optional[float] = None
        self._WKB_violation_efolds_subh: Optional[float] = None

    def __enter__(self):
        super().__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)

    def message(self, current_u, msg):
        current_time = time.time()
        since_last_notify = current_time - self._last_notify
        since_start = current_time - self._start_time

        update_number = self.report_notify()

        u_complete = self._u_target - current_u
        u_remain = self._u_range - u_complete
        percent_remain = u_remain / self._u_range
        print(
            f"** STATUS UPDATE #{update_number}: {self._label} WKB Q_k(z) integration for k = {self._k.k_inv_Mpc:.5g}/Mpc (store_id={self._k.store_id}) has been running for {format_time(since_start)} ({format_time(since_last_notify)} since last notification)"
        )
        print(
            f"|    current u={current_u:.5g} (initial u={self._u_init:.5g}, target z={self._u_target:.5g}, z complete={u_complete:.5g}, z remain={u_remain:.5g}, {percent_remain:.3%} remains)"
        )
        if self._last_u is not None:
            u_delta = current_u - self._last_u
            print(f"|    u advance since last update: Delta u = {u_delta:.5g}")
        print(
            f"|    {self.RHS_evaluations} RHS evaluations, mean {self.mean_RHS_time:.5g}s per evaluation, min RHS time = {self.min_RHS_time:.5g}s, max RHS time = {self.max_RHS_time:.5g}s"
        )
        print(f"|    {msg}")

        self._last_u = current_u

    def update_Q(self, Q: float):
        if self._largest_Q is None or Q > self._largest_Q:
            self._largest_Q = Q

        if self._smallest_Q is None or Q < self._smallest_Q:
            self._smallest_Q = Q

    def report_WKB_violation(self, z: float, efolds_subh: float):
        if self._WKB_violation:
            return

        print(
            f"!! WARNING: {self._label} WKB Q_k(z) integration for k = {self._k.k_inv_Mpc:.5g}/Mpc (store_id={self._k.store_id}) may have violated the validity criterion for the WKB approximation"
        )
        print(f"|    current z={z:.5g}, e-folds inside horizon={efolds_subh:.3g}")
        self._WKB_violation = True
        self._WKB_violation_z = z
        self._WKB_violation_efolds_subh = efolds_subh

    @property
    def has_WKB_violation(self) -> bool:
        return self._WKB_violation

    @property
    def WKB_violation_z(self) -> Optional[float]:
        return self._WKB_violation_z

    @property
    def WKB_violation_efolds_subh(self) -> Optional[float]:
        return self._WKB_violation_efolds_subh

    @property
    def largest_Q(self) -> Optional[float]:
        return self._largest_Q

    @property
    def smallest_Q(self) -> Optional[float]:
        return self._smallest_Q
