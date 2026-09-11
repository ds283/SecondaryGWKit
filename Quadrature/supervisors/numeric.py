import time
from math import log
from typing import Optional, Union

from CosmologyConcepts import wavenumber, redshift
from Quadrature.supervisors.base import IntegrationSupervisor, DEFAULT_UPDATE_INTERVAL
from utilities import format_time


# ``delta_logz`` is supplied as a spacing in log10(1+z); converting it to a spacing in
# ln(1+z) -- which is what multiplying by (1+z) produces a Delta z from -- costs a factor ln 10
LN_10 = log(10.0)


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
        """
        **Superseded.** Prompt 11 of ``prompts/GkTk-remedial`` moved the oscillation-resolution
        test out of the ODE right-hand side and on to the *returned sample grid*, where
        ``numeric_with_phase_cut`` now performs it after the solve
        (``Quadrature/integrators/numeric_with_phase_cut.py``,
        ``scan_sample_grid_for_unresolved_osc``). Neither ``GkNumericIntegration.RHS`` nor
        ``TkNumericIntegration.RHS`` calls this method any more, and the flag those integrators
        return no longer comes from this class.

        The method is retained because it is the only per-step form of the test, and because
        removing it would silently change any future caller that still uses it. Its unit slip is
        fixed here: ``delta_logz`` is a spacing in ``log10(1+z)`` -- that is what ``main.py``
        passes, ``1/source_samples_per_log10z`` at ``:630`` and ``:1199`` -- so the corresponding
        spacing in z is ``(1+z) * delta_logz * ln(10)``, not ``(1+z) * delta_logz``. The missing
        ``ln 10 = 2.303`` understated the grid spacing and moved the trip point from the intended
        x ~ 273 to x ~ 630 (review §10.2, §13.1).

        :param z: current redshift
        :param wavelength: local oscillation wavelength, as a Delta z
        :param efolds_subh: e-folds inside the horizon at this redshift
        """
        if self._has_unresolved_osc:
            return

        if self._delta_logz is None:
            return

        grid_spacing = (1.0 + z) * self._delta_logz * LN_10
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
