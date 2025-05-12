import time
from typing import Union

from CosmologyConcepts import wavenumber, redshift
from Quadrature.supervisors.base import IntegrationSupervisor
from utilities import format_time

DEFAULT_KEXIT_NOTIFY_INTERVAL = 2 * 60


class WavenumberExitSupervisor(IntegrationSupervisor):
    def __init__(
        self,
        k: wavenumber,
        z_init: Union[redshift, float],
        notify_interval=DEFAULT_KEXIT_NOTIFY_INTERVAL,
    ):
        super().__init__(notify_interval)

        self._k: wavenumber = k
        self._z_init: float = (
            z_init.z if isinstance(z_init, redshift) else float(z_init)
        )
        self._last_z = self._z_init

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

        print(
            f"** STATUS UPDATE #{update_number}: horizon-exit finding integration for k = {self._k.k_inv_Mpc:.5g}/Mpc (store_id={self._k.store_id}) has been running for {format_time(since_start)} ({format_time(since_last_notify)} since last notification)"
        )
        print(f"|    current z={current_z:.5g} (init z={self._z_init:.5g})")
        if self._last_z is not None:
            z_delta = self._last_z - current_z
            print(f"|    redshift advance since last update: Delta z = {z_delta:.5g}")
        print(
            f"|    {self.RHS_evaluations} RHS evaluations, mean {self.mean_RHS_time:.5g}s per evaluation, min RHS time = {self.min_RHS_time:.5g}s, max RHS time = {self.max_RHS_time:.5g}s"
        )
        print(f"|    {msg}")

        self._last_z = current_z
