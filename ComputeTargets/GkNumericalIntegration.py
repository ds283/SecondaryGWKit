import time
from typing import Optional, List

import ray
from math import fabs, pi, log, sqrt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

from ComputeTargets.BackgroundModel import BackgroundModel
from ComputeTargets.WKB_tensor_Green import WKB_omegaEff_sq, WKB_d_ln_omegaEffPrime_dz
from ComputeTargets.analytic_Gk import compute_analytic_G, compute_analytic_Gprime
from ComputeTargets.integration_metadata import IntegrationSolver
from ComputeTargets.integration_supervisor import (
    IntegrationSupervisor,
    DEFAULT_UPDATE_INTERVAL,
    RHS_timer,
)
from CosmologyConcepts import wavenumber, redshift, redshift_array, wavenumber_exit_time
from Datastore import DatastoreObject
from MetadataConcepts import tolerance, store_tag
from defaults import (
    DEFAULT_ABS_TOLERANCE,
    DEFAULT_REL_TOLERANCE,
    DEFAULT_FLOAT_PRECISION,
)
from utilities import check_units, format_time, IntegrationData

# RHS of ODE system
#
# State layout:
#   state[0] = G_k(z, z')
#   state[1] = Gprime_k(z, z')
G_INDEX = 0
GPRIME_INDEX = 1
EXPECTED_SOL_LENGTH = 2


class GkIntegrationSupervisor(IntegrationSupervisor):
    def __init__(
        self,
        k: wavenumber,
        z_source: redshift,
        z_final: redshift,
        notify_interval: int = DEFAULT_UPDATE_INTERVAL,
        delta_logz: Optional[float] = None,
    ):
        super().__init__(notify_interval)

        self._k: wavenumber = k
        self._z_source: float = z_source.z
        self._z_final: float = z_final.z

        self._z_range: float = self._z_source - self._z_final

        self._last_z: float = self._z_source

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

        z_complete = self._z_source - current_z
        z_remain = self._z_range - z_complete
        percent_remain = 100.0 * (z_remain / self._z_range)
        print(
            f"** STATUS UPDATE #{update_number}: Integration for Gr(k) for k = {self._k.k_inv_Mpc:.5g}/Mpc (store_id={self._k.store_id}) has been running for {format_time(since_start)} ({format_time(since_last_notify)} since last notification)"
        )
        print(
            f"|    current z={current_z:.5g} (source z={self._z_source:.5g}, target z={self._z_final:.5g}, z complete={z_complete:.5g}, z remain={z_remain:.5g}, {percent_remain:.3g}% remains)"
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
                f"!! WARNING: Integration for Gr_k(z, z') for k = {self._k.k_inv_Mpc:.5g}/Mpc (store_id={self._k.store_id}) may have developed unresolved oscillations"
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


def search_G_minimum(sol, start_z: float, stop_z: float):
    start_Gprime = sol(start_z)[GPRIME_INDEX]
    last_Gprime = start_Gprime

    stepsize = -(1e-3) * start_z
    found_zero = False

    last_z = start_z
    current_z = start_z + stepsize
    current_Gprime = None
    while current_z > stop_z:
        current_Gprime = sol(current_z)[GPRIME_INDEX]

        # has there been a sign change since the last time we sampled Gprime?
        if current_Gprime * last_Gprime < 0 and last_Gprime < 0:
            found_zero = True
            break

        last_z = current_z

        stepsize = -(1e-3) * current_z
        current_z += stepsize
        last_Gprime = current_Gprime

    if not found_zero:
        raise RuntimeError(
            f"Did not find zero of Gprime in the search window (start_z={start_z:.5g}, stop_z={stop_z:.5g}), current_z={current_z:.5g}, start Gprime={start_Gprime:.5g}, last Gprime={last_Gprime:.5g}, current Gprime={current_Gprime:.5g}"
        )

    root = root_scalar(
        lambda z: sol(z)[GPRIME_INDEX],
        bracket=(last_z, current_z),
        xtol=1e-6,
        rtol=1e-4,
    )

    if not root.converged:
        raise RuntimeError(
            f'root_scalar() did not converge to a solution: z_bracket=({last_z:.5g}, {current_z:.5g}), iterations={root.iterations}, method={root.method}: "{root.flag}"'
        )

    root_z = root.root
    sol_root = sol(root_z)

    return {
        "z": root_z,
        "G": sol_root[G_INDEX],
        "Gprime": sol_root[GPRIME_INDEX],
    }


@ray.remote
def compute_Gk(
    model: BackgroundModel,
    k: wavenumber_exit_time,
    z_source: redshift,
    z_sample: redshift_array,
    atol: float = DEFAULT_ABS_TOLERANCE,
    rtol: float = DEFAULT_REL_TOLERANCE,
    delta_logz: Optional[float] = None,
    mode: str = None,
    stop_search_window_z_begin: Optional[float] = None,
    stop_search_window_z_end: Optional[float] = None,
) -> dict:
    k_wavenumber: wavenumber = k.k
    check_units(k_wavenumber, model.cosmology)

    mode = mode.lower()
    if mode is not None and mode not in ["stop"]:
        raise ValueError(f'compute_Gk: unknown compute mode "{mode}"')

    if mode in ["stop"]:
        if stop_search_window_z_begin is None:
            raise ValueError(
                "compute_Gk: in 'stop' mode, stop_search_window_z_begin must be specified"
            )
        if stop_search_window_z_end is None:
            raise ValueError(
                "compute_Gk: in 'stop' mode, stop_search_window_z_end must be specified"
            )

        if stop_search_window_z_begin < stop_search_window_z_end:
            stop_search_window_z_begin, stop_search_window_z_end = (
                stop_search_window_z_end,
                stop_search_window_z_begin,
            )
            print(
                f"## compute_Gk: stop search window start and end arguments in the wrong order (for k={k.k.k_inv_Mpc:.5g}/Mpc). Now searching in interval: z in ({stop_search_window_z_begin}, {stop_search_window_z_end})"
            )

        max_z = z_sample.max.z
        min_z = z_sample.min.z
        if stop_search_window_z_begin > max_z:
            raise ValueError(
                f"compute_Gk: (for k={k.k.k_inv_Mpc:.5g}/Mpc) specified 'stop' window starting redshift z={stop_search_window_z_begin:.5g} exceeds highest z-response sample point z={max_z:.5g}"
            )
        if stop_search_window_z_end < min_z:
            raise ValueError(
                f"## compute_Gk: (for k={k.k.k_inv_Mpc:.5g}/Mpc) specified 'stop' window ending redshift z={stop_search_window_z_end:.5g} is smaller than lowest z-response sample point z={min_z:.5g}. Search will terminate at z={min_z:.5g}."
            )

        if (
            fabs(stop_search_window_z_begin - stop_search_window_z_end)
            < DEFAULT_FLOAT_PRECISION
        ):
            raise ValueError(
                f"## compute_Gk: (for k={k.k.k_inv_Mpc:.5g}/Mpc) specified search window has zero extent"
            )

        if (
            fabs(z_source.z - z_sample.min.z) < DEFAULT_ABS_TOLERANCE
            or z_source.store_id == z_sample.min.store_id
        ):
            raise ValueError(
                f"## compute_Gk: (for k={k.k.k_inv_Mpc:.5g}/Mpc) in 'stop' mode, the source redshift and the lowest response redshift cannot be equal"
            )

    # obtain dimensionful value of wavenumber; this should be measured in the same units used by the cosmology
    k_float = k_wavenumber.k

    z_min = float(z_sample.min)

    def RHS(z, state, supervisor) -> List[float]:
        """
        k *must* be measured using the same units used for H(z) in the cosmology, otherwise we will not get
        correct dimensionless ratios
        """
        with RHS_timer(supervisor) as timer:
            G = state[G_INDEX]
            Gprime = state[GPRIME_INDEX]

            if supervisor.notify_available:
                supervisor.message(
                    z,
                    f"current state: Gr(k) = {G:.5g}, dGr(k)/dz = {Gprime:.5g}",
                )
                supervisor.reset_notify_time()

            H = model.functions.Hubble(z)
            eps = model.functions.epsilon(z)

            one_plus_z = 1.0 + z
            one_plus_z_2 = one_plus_z * one_plus_z

            dG_dz = Gprime

            k_over_H = k_float / H
            k_over_H_2 = k_over_H * k_over_H

            dGprime_dz = (
                -(eps / one_plus_z) * Gprime
                - (k_over_H_2 + (eps - 2.0) / one_plus_z_2) * G
            )

            # try to detect how many oscillations will fit into the log-z grid
            # spacing
            # If the grid spacing is smaller than the oscillation wavelength, then
            # evidently we cannot resolve the oscillations
            omega_WKB_sq = WKB_omegaEff_sq(model, k_float, z)

            if omega_WKB_sq > 0.0:
                wavelength = 2.0 * pi / sqrt(omega_WKB_sq)
                supervisor.report_wavelength(z, wavelength, log((1.0 + z) * k_over_H))

        return [dG_dz, dGprime_dz]

    with GkIntegrationSupervisor(
        k_wavenumber, z_source, z_sample.min, delta_logz=delta_logz
    ) as supervisor:
        # initial conditions should be
        #   G(z', z') = 0
        #   Gprime(z' z') = -1/(a0 H(z'))
        # however we would rather not have a delicate initial condition for Gprime, so we
        # instead solve with the boundary conditions Gprime = -1 and rescale later
        initial_state = [0.0, 1.0]

        if mode == "stop":
            # set up an event to terminate the integration when 6 e-folds inside the horizon
            def stop_event(z, state, supervisor):
                return z - stop_search_window_z_end + DEFAULT_FLOAT_PRECISION

            # terminate integration when > 5 e-folds inside the horizon
            stop_event.terminal = True

            events = [stop_event]
            dense_output = True
        else:
            events = None
            dense_output = False

        sol = solve_ivp(
            RHS,
            method="RK45",
            t_span=(z_source.z, z_min),
            y0=initial_state,
            t_eval=z_sample.as_float_list(),
            events=events,
            dense_output=dense_output,
            atol=atol,
            rtol=rtol,
            args=(supervisor,),
        )

    # test whether the integration concluded successfully
    if not sol.success:
        raise RuntimeError(
            f'compute_Gk: integration did not terminate successfully (k={k_wavenumber.k_inv_Mpc}/Mpc, z_source={z_source.z}, error at z={sol.t[-1]}, "{sol.message}")'
        )

    if mode == "stop" and sol.status != 1:
        raise RuntimeError(
            f'compute_Gk: mode is "{mode}", but solution did not terminate due to an event"'
        )

    sampled_z = sol.t
    sampled_values = sol.y
    if len(sampled_values) != EXPECTED_SOL_LENGTH:
        raise RuntimeError(
            f"compute_Gk: solution does not have expected number of members (expected {EXPECTED_SOL_LENGTH}, found {len(sampled_values)}; k={k_wavenumber.k.k_inv_Mpc}/Mpc, length of sol.t={len(sampled_z)})"
        )
    sampled_G = sampled_values[G_INDEX]
    sampled_Gprime = sampled_values[GPRIME_INDEX]

    # if no data points returned, check if this is because the target z (ie., lowest z_response)
    # and the source z agree.
    # If so, then we know the correct value from the initial data.
    if (
        len(sampled_z) == 0
        and len(z_sample) == 0
        and (
            fabs(z_source.z - z_sample.min.z) < DEFAULT_ABS_TOLERANCE
            or z_source.store_id == z_sample.min.store_id
        )
    ):
        sampled_z.append(z_source.z)
        sampled_G.append(0.0)
        sampled_Gprime.append(1.0)

    returned_values = sampled_z.size
    if mode != "stop":
        expected_values = len(z_sample)

        if returned_values != expected_values:
            raise RuntimeError(
                f"compute_Gk: solve_ivp returned {returned_values} samples, but expected {expected_values}"
            )

    stop_deltaz_subh = None
    stop_G = None
    stop_Gprime = None

    if mode == "stop":
        payload = search_G_minimum(
            sol.sol, start_z=stop_search_window_z_begin, stop_z=stop_search_window_z_end
        )
        stop_deltaz_subh = k.z_exit - payload["z"]
        stop_G = payload["G"]
        stop_Gprime = payload["Gprime"]

    # validate that the samples of the solution correspond to the z-sample points that we specified.
    # This really should be true, but there is no harm in being defensive.
    for i in range(returned_values):
        diff = sampled_z[i] - z_sample[i].z
        if fabs(diff) > DEFAULT_ABS_TOLERANCE:
            raise RuntimeError(
                f"compute_Gk: solve_ivp returned sample points that differ from those requested (difference={diff} at i={i})"
            )

    return {
        "data": IntegrationData(
            compute_time=supervisor.integration_time,
            compute_steps=int(sol.nfev),
            RHS_evaluations=supervisor.RHS_evaluations,
            mean_RHS_time=supervisor.mean_RHS_time,
            max_RHS_time=supervisor.max_RHS_time,
            min_RHS_time=supervisor.min_RHS_time,
        ),
        "G_sample": sampled_G,
        "Gprime_sample": sampled_Gprime,
        "solver_label": "solve_ivp+RK45-stepping0",
        "has_unresolved_osc": supervisor.has_unresolved_osc,
        "unresolved_z": supervisor.unresolved_z,
        "unresolved_efolds_subh": supervisor.unresolved_efolds_subh,
        "stop_deltaz_subh": stop_deltaz_subh,
        "stop_G": stop_G,
        "stop_Gprime": stop_Gprime,
    }


class GkNumericalIntegration(DatastoreObject):
    """
    Encapsulates all sample points produced during a single integration of the
    tensor Green's function, labelled by a wavenumber k, and two redshifts:
    the source redshift, and the response redshift.
    A single integration fixes the source redshift and determines the Green function
    as a function of the response redshift.
    However, once these have been computed and cached, we can obtain the result as
    a function of the source redshift if we wish
    """

    def __init__(
        self,
        payload,
        solver_labels: dict,
        model: BackgroundModel,
        k: wavenumber_exit_time,
        atol: tolerance,
        rtol: tolerance,
        z_source: Optional[redshift] = None,
        z_sample: Optional[redshift_array] = None,
        label: Optional[str] = None,
        tags: Optional[List[store_tag]] = None,
        delta_logz: Optional[float] = None,
        mode: Optional[str] = None,
    ):
        k_wavenumber: wavenumber = k.k
        check_units(k_wavenumber, model.cosmology)

        self._solver_labels = solver_labels
        self._delta_logz = delta_logz
        self._mode = mode.lower() if mode is not None else None

        if self._mode is not None and self._mode not in ["stop"]:
            raise ValueError(
                f'GkNumericalIntegration: unknown compute mode "{self._mode}"'
            )

        self._stop_search_window_start_attr = "z_exit_subh_e3"
        self._stop_search_window_end_attr = "z_exit_subh_e6"

        self._z_sample = z_sample
        if payload is None:
            DatastoreObject.__init__(self, None)
            self._data = None
            self._compute_time = None
            self._compute_steps = None
            self._RHS_evaluations = None
            self._mean_RHS_time = None
            self._max_RHS_time = None
            self._min_RHS_time = None

            self._has_unresolved_osc = None
            self._unresolved_z = None
            self._unresolved_efolds_subh = None

            self._init_efolds_suph = None
            self._stop_deltaz_subh = None
            self._stop_G = None
            self._stop_Gprime = None

            self._solver = None

            self._values = None
        else:
            DatastoreObject.__init__(self, payload["store_id"])
            self._data = payload["data"]

            self._has_unresolved_osc = payload["has_unresolved_osc"]
            self._unresolved_z = payload["unresolved_z"]
            self._unresolved_efolds_subh = payload["unresolved_efolds_subh"]

            self._init_efolds_suph = payload["init_efolds_suph"]
            self._stop_deltaz_subh = payload["stop_deltaz_subh"]
            self._stop_G = payload["stop_G"]
            self._stop_Gprime = payload["stop_Gprime"]

            self._solver = payload["solver"]

            self._values = payload["values"]

        # check that all sample points are *later* than the specified source redshift
        if z_source is not None and self._z_sample is not None:
            z_source_float = float(z_source)
            for z in self._z_sample:
                z_float = float(z)
                if z_float > z_source_float:
                    raise ValueError(
                        f"Redshift sample point z={z_float:.5g} exceeds source redshift z={z_source_float:.5g}"
                    )

        # store parameters
        self._label = label
        self._tags = tags if tags is not None else []

        self._model = model

        self._k_exit = k
        self._z_source = z_source

        self._compute_ref = None

        self._atol = atol
        self._rtol = rtol

    @property
    def model(self) -> BackgroundModel:
        return self._model

    @property
    def k(self) -> wavenumber:
        return self._k_exit.k

    @property
    def z_exit(self) -> float:
        return self._k_exit.z_exit

    @property
    def label(self) -> str:
        return self._label

    @property
    def tags(self) -> List[store_tag]:
        return self._tags

    @property
    def z_source(self):
        return self._z_source

    @property
    def z_sample(self):
        return self._z_sample

    @property
    def data(self) -> IntegrationData:
        if self.values is None:
            raise RuntimeError("values have not yet been populated")

        return self._data

    @property
    def has_unresolved_osc(self) -> bool:
        if self._has_unresolved_osc is None:
            raise RuntimeError("has_unresolved_osc has not yet been populated")
        return self._has_unresolved_osc

    @property
    def unresolved_z(self) -> float:
        if self._has_unresolved_osc is None:
            raise RuntimeError("has_unresolved_osc has not yet been populated")

        return self._unresolved_z

    @property
    def unresolved_efolds_subh(self) -> float:
        if self._has_unresolved_osc is None:
            raise RuntimeError("has_unresolved_osc has not yet been populated")

        return self._unresolved_efolds_subh

    @property
    def init_efolds_suph(self) -> float:
        if self._init_efolds_suph is None:
            raise RuntimeError("init_efolds_suph has not yet been populated")

        return self._init_efolds_suph

    @property
    def stop_deltaz_subh(self) -> float:
        if self._stop_deltaz_subh is None:
            raise RuntimeError("stop_deltaz_subh has not yet been populated")

        return self._stop_deltaz_subh

    @property
    def stop_G(self) -> float:
        if self._stop_G is None:
            raise RuntimeError("stop_G has not yet been populated")

        return self._stop_G

    @property
    def stop_Gprime(self) -> float:
        if self._stop_Gprime is None:
            raise RuntimeError("stop_Gprime has not yet been populated")

        return self._stop_Gprime

    @property
    def solver(self) -> IntegrationSolver:
        if self._solver is None:
            raise RuntimeError("solver has not yet been populated")
        return self._solver

    @property
    def values(self) -> List:
        if hasattr(self, "_do_not_populate"):
            raise RuntimeError(
                "GkNumericalIntegration: values read but _do_not_populate is set"
            )

        if self._values is None:
            raise RuntimeError("values has not yet been populated")
        return self._values

    def compute(self, label: Optional[str] = None):
        if hasattr(self, "_do_not_populate"):
            raise RuntimeError(
                "GkNumericalIntegration: compute() called but _do_not_populate is set"
            )

        if self._values is not None:
            raise RuntimeError("values have already been computed")

        if self._z_source is None or self._z_sample is None:
            raise RuntimeError(
                "Object has not been configured correctly for a concrete calculation (z_source or z_sample is missing). It can only represent a query."
            )

        # check that source redshift is not too far inside the horizon for this k-mode
        z_limit = self._k_exit.z_exit_subh_e3
        if self._z_source.z < z_limit - DEFAULT_FLOAT_PRECISION:
            raise ValueError(
                f"Specified source redshift z_source={self._z_source.z:.5g} is more than 3-efolds inside the horizon for k={self._k_exit.k.k_inv_Mpc:.5g}/Mpc (horizon re-entry at z_entry={self._k_exit.z_exit:.5g})"
            )

        # replace label if specified
        if label is not None:
            self._label = label

        payload = {}
        if self._mode in ["stop"]:
            payload["mode"] = self._mode
            payload["stop_search_window_z_begin"] = getattr(
                self._k_exit, self._stop_search_window_start_attr
            )
            payload["stop_search_window_z_end"] = getattr(
                self._k_exit, self._stop_search_window_end_attr
            )

        self._compute_ref = compute_Gk.remote(
            self._model,
            self._k_exit,
            self.z_source,
            self.z_sample,
            atol=self._atol.tol,
            rtol=self._rtol.tol,
            delta_logz=self._delta_logz,
            **payload,
        )
        return self._compute_ref

    def store(self) -> Optional[bool]:
        if self._compute_ref is None:
            raise RuntimeError(
                "GkNumericalIntegration: store() called, but no compute() is in progress"
            )

        # check whether the computation has actually resolved
        resolved, unresolved = ray.wait([self._compute_ref], timeout=0)

        # if not, return None
        if len(resolved) == 0:
            return None

        # retrieve result and populate ourselves
        data = ray.get(self._compute_ref)
        self._compute_ref = None

        self._data = data["data"]

        self._has_unresolved_osc = data["has_unresolved_osc"]
        self._unresolved_z = data["unresolved_z"]
        self._unresolved_efolds_subh = data["unresolved_efolds_subh"]

        self._stop_deltaz_subh = data.get("stop_deltaz_subh", None)
        self._stop_G = data.get("stop_G", None)
        self._stop_Gprime = data.get("stop_Gprime", None)

        Hsource = self._model.functions.Hubble(self.z_source.z)
        k_over_aH = (1.0 + self.z_source.z) * self.k.k / Hsource
        self._init_efolds_suph = -log(k_over_aH)

        G_sample = data["G_sample"]
        Gprime_sample = data["Gprime_sample"]

        tau_source = self._model.functions.tau(self._z_sample[0].z)

        # need to be aware that G_sample may not be as long as self._z_sample, if we are working in "stop" mode
        self._values = []
        for i in range(len(G_sample)):
            current_z = self._z_sample[i]
            current_z_float = current_z.z
            H = self._model.functions.Hubble(current_z_float)
            tau = self._model.functions.tau(current_z_float)

            analytic_G = compute_analytic_G(
                self.k.k, 1.0 / 3.0, tau_source, tau, Hsource
            )
            analytic_Gprime = compute_analytic_Gprime(
                self.k.k, 1.0 / 3.0, tau_source, tau, Hsource, H
            )
            omega_WKB_sq = WKB_omegaEff_sq(self._model, self.k.k, current_z_float)
            WKB_criterion = fabs(
                WKB_d_ln_omegaEffPrime_dz(self._model, self.k.k, current_z_float)
            ) / sqrt(fabs(omega_WKB_sq))

            # create new GkNumericalValue object
            self._values.append(
                GkNumericalValue(
                    None,
                    current_z,
                    G_sample[i],
                    Gprime_sample[i],
                    analytic_G=analytic_G,
                    analytic_Gprime=analytic_Gprime,
                    omega_WKB_sq=omega_WKB_sq,
                    WKB_criterion=WKB_criterion,
                )
            )

        self._solver = self._solver_labels[data["solver_label"]]

        return True


class GkNumericalValue(DatastoreObject):
    """
    Encapsulates a single sampled value of the tensor Green's transfer functions.
    Parameters such as wavenumber k, source redshift z_source, etc., are held by the
    owning GkNumericalIntegration object
    """

    def __init__(
        self,
        store_id: int,
        z: redshift,
        G: float,
        Gprime: float,
        analytic_G: Optional[float] = None,
        analytic_Gprime: Optional[float] = None,
        omega_WKB_sq: Optional[float] = None,
        WKB_criterion: Optional[float] = None,
    ):
        DatastoreObject.__init__(self, store_id)

        self._z = z
        self._G = G
        self._Gprime = Gprime

        self._analytic_G = analytic_G
        self._analytic_Gprime = analytic_Gprime

        self._omega_WKB_sq = omega_WKB_sq
        self._WKB_criterion = WKB_criterion

    def __float__(self):
        """
        Cast to float. Returns value of the Green's function
        :return:
        """
        return self.G

    @property
    def z(self) -> redshift:
        return self._z

    @property
    def G(self) -> float:
        return self._G

    @property
    def Gprime(self) -> float:
        return self._Gprime

    @property
    def analytic_G(self) -> Optional[float]:
        return self._analytic_G

    @property
    def analytic_Gprime(self) -> Optional[float]:
        return self._analytic_Gprime

    @property
    def omega_WKB_sq(self) -> Optional[float]:
        return self._omega_WKB_sq

    @property
    def WKB_criterion(self) -> Optional[float]:
        return self._WKB_criterion
