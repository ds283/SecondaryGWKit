import time
from typing import Optional, Union

from CosmologyConcepts import wavenumber, redshift
from Quadrature.supervisors.base import IntegrationSupervisor, DEFAULT_UPDATE_INTERVAL
from utilities import format_time


class NumericIntegrationSupervisor(IntegrationSupervisor):
    def __init__(
        self,
        k: wavenumber,
        z_init: Union[redshift, float],
        z_final: Union[redshift, float],
        label: str,
        notify_interval: int = DEFAULT_UPDATE_INTERVAL,
        delta_logz: Optional[float] = None,
    ):
        super().__init__(notify_interval)

        self._label: str = label

        self._k: wavenumber = k
        self._z_init: float = (
            z_init.z if isinstance(z_init, redshift) else float(z_init)
        )
        self._z_final: float = (
            z_final.z if isinstance(z_final, redshift) else float(z_final)
        )

        self._z_range: float = self._z_init - self._z_final

        self._last_z: float = self._z_init

        self._has_unresolved_osc: bool = False
        self._delta_logz: float = delta_logz
        self._unresolved_osc_z: Optional[float] = None
        self._unresolved_osc_efolds_subh: Optional[float] = None

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
            f"** STATUS UPDATE #{update_number}: {self._label} integration for k = {self._k.k_inv_Mpc:.5g}/Mpc (store_id={self._k.store_id}) has been running for {format_time(since_start)} ({format_time(since_last_notify)} since last notification)"
        )
        print(
            f"|    current z={current_z:.5g} (init z={self._z_init:.5g}, target z={self._z_final:.5g}, z complete={z_complete:.5g}, z remain={z_remain:.5g}, {percent_remain:.3%} remains)"
        )
        if self._last_z is not None:
            z_delta = self._last_z - current_z
            print(f"|    redshift advance since last update: Delta z = {z_delta:.5g}")
        print(
            f"|    {self.RHS_evaluations} RHS evaluations, mean {self.mean_RHS_time:.5g}s per evaluation, min RHS time = {self.min_RHS_time:.5g}s, max RHS time = {self.max_RHS_time:.5g}s"
        )
        print(f"|    {msg}")

        self._last_z = current_z

    def report_wavelength(self, z: float, wavelength: float, efolds_subh: float):
        if self._has_unresolved_osc:
            return

        if self._delta_logz is None:
            return

        grid_spacing = (1.0 + z) * self._delta_logz
        if wavelength < grid_spacing:
            print(
                f"!! WARNING: {self._label} integration for k = {self._k.k_inv_Mpc:.5g}/Mpc (store_id={self._k.store_id}) may have developed unresolved oscillations"
            )
            print(
                f"|    current z={z:.5g}, e-folds inside horizon={efolds_subh:.3g} | approximate wavelength Delta z={wavelength:.5g}, approximate grid spacing at this z: {grid_spacing:.5g}"
            )
            self._has_unresolved_osc = True
            self._unresolved_osc_z = z
            self._unresolved_osc_efolds_subh = efolds_subh

    @property
    def has_unresolved_osc(self):
        if self._delta_logz is None:
            return None

        return self._has_unresolved_osc

    @property
    def unresolved_z(self):
        if self._has_unresolved_osc is False or self._delta_logz is None:
            return None

        return self._unresolved_osc_z

    @property
    def unresolved_efolds_subh(self):
        if self._has_unresolved_osc is False or self._delta_logz is None:
            return None

        return self._unresolved_osc_efolds_subh
