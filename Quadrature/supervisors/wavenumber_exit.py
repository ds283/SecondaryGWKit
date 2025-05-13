import time

from CosmologyConcepts import wavenumber
from Quadrature.supervisors.base import IntegrationSupervisor
from utilities import format_time

DEFAULT_KEXIT_NOTIFY_INTERVAL = 2 * 60


class WavenumberExitSupervisor(IntegrationSupervisor):

    def __init__(
        self,
        k: wavenumber,
        log_z_init: float,
        log_z_final: float,
        notify_interval=DEFAULT_KEXIT_NOTIFY_INTERVAL,
    ):
        super().__init__(notify_interval)

        self._k: wavenumber = k

        self._log_z_init: float = log_z_init
        self._log_z_final: float = log_z_final
        self._log_z_range: float = self._log_z_init - self._log_z_final

        self._last_log_z = self._log_z_init

    def __enter__(self):
        super().__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)

    def message(self, current_log_z, msg):
        current_time = time.time()
        since_last_notify = current_time - self._last_notify
        since_start = current_time - self._start_time

        update_number = self.report_notify()

        log_z_complete = self._log_z_init - current_log_z
        log_z_remain = self._log_z_range - log_z_complete
        percent_remain = log_z_remain / self._log_z_range
        print(
            f"** STATUS UPDATE #{update_number}: horizon-exit finding integration for k = {self._k.k_inv_Mpc:.5g}/Mpc (store_id={self._k.store_id}) has been running for {format_time(since_start)} ({format_time(since_last_notify)} since last notification)"
        )
        print(
            f"|    current log(1+z)={current_log_z:.5g} (init log(1+z)z={self._log_z_init:.5g}, target log(1+z)={self._log_z_final:.5g}, log(1+z)-complete={log_z_complete:.5g}, log(1+z)-remain={log_z_remain:.5g}, {percent_remain:.3%} remains)"
        )
        if self._last_log_z is not None:
            log_z_delta = self._last_log_z - current_log_z
            print(
                f"|    redshift advance since last update: Delta log(1+z) = {log_z_delta:.5g}"
            )
        print(
            f"|    {self.RHS_evaluations} RHS evaluations, mean {self.mean_RHS_time:.5g}s per evaluation, min RHS time = {self.min_RHS_time:.5g}s, max RHS time = {self.max_RHS_time:.5g}s"
        )
        print(f"|    {msg}")

        self._last_log_z = current_log_z
