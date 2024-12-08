import time
from math import fabs, pi, log, sqrt
from typing import Optional, List

import ray
from scipy.integrate import solve_ivp

from ComputeTargets.BackgroundModel import BackgroundModel, ModelProxy
from ComputeTargets.WKB_Tk import Tk_omegaEff_sq, Tk_d_ln_omegaEffPrime_dz
from ComputeTargets.analytic_Tk import compute_analytic_T, compute_analytic_Tprime
from CosmologyConcepts import redshift_array, wavenumber, redshift, wavenumber_exit_time
from Datastore import DatastoreObject
from LiouvilleGreen.integration_tools import find_phase_minimum
from MetadataConcepts import tolerance, store_tag
from Quadrature.integration_metadata import IntegrationSolver, IntegrationData
from Quadrature.integration_supervisor import (
    DEFAULT_UPDATE_INTERVAL,
    IntegrationSupervisor,
    RHS_timer,
)
from Units import check_units
from defaults import (
    DEFAULT_ABS_TOLERANCE,
    DEFAULT_REL_TOLERANCE,
    DEFAULT_FLOAT_PRECISION,
)
from utilities import format_time

# RHS of ODE system
#
# State layout:
#   state[0] = a_0 tau(z) [tau = conformal time]
#   state[1] = T(z)
#   state[2] = dT/dz = T' = "T prime"
T_INDEX = 0
TPRIME_INDEX = 1
EXPECTED_SOL_LENGTH = 2


class TkIntegrationSupervisor(IntegrationSupervisor):
    def __init__(
        self,
        k: wavenumber,
        z_init: redshift,
        z_final: redshift,
        notify_interval: int = DEFAULT_UPDATE_INTERVAL,
        delta_logz: Optional[float] = None,
    ):
        super().__init__(notify_interval)

        self._k: wavenumber = k
        self._z_init: float = z_init.z
        self._z_final: float = z_final.z

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
            f"** STATUS UPDATE #{update_number}: Integration for T(k) for k = {self._k.k_inv_Mpc:.5g}/Mpc (store_id={self._k.store_id}) has been running for {format_time(since_start)} ({format_time(since_last_notify)} since last notification)"
        )
        print(
            f"|    current z={current_z:.5g} (initial z={self._z_init:.5g}, target z={self._z_final:.5g}, z complete={z_complete:.5g}, z remain={z_remain:.5g}, {percent_remain:.3%} remains)"
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
                f"!! WARNING: Integration for T_k(z) for k = {self._k.k_inv_Mpc:.5g}/Mpc (store_id={self._k.store_id}) may have developed unresolved oscillations"
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


@ray.remote
def compute_Tk(
    model_proxy: ModelProxy,
    k: wavenumber_exit_time,
    z_sample: redshift_array,
    z_init: redshift,
    atol: float = DEFAULT_ABS_TOLERANCE,
    rtol: float = DEFAULT_REL_TOLERANCE,
    delta_logz: Optional[float] = None,
    mode: str = None,
    stop_search_window_z_begin: Optional[float] = None,
    stop_search_window_z_end: Optional[float] = None,
) -> dict:
    k_wavenumber: wavenumber = k.k
    check_units(k_wavenumber, model_proxy)

    model: BackgroundModel = model_proxy.get()

    mode = mode.lower()
    if mode is not None and mode not in ["stop"]:
        raise ValueError(f'compute_Gk: unknown compute mode "{mode}"')

    if mode in ["stop"]:
        if stop_search_window_z_begin is None:
            raise ValueError(
                "compute_Tk: in 'stop' mode, stop_search_window_z_begin must be specified"
            )
        if stop_search_window_z_end is None:
            raise ValueError(
                "compute_Tk: in 'stop' mode, stop_search_window_z_end must be specified"
            )

        if stop_search_window_z_begin < stop_search_window_z_end:
            stop_search_window_z_begin, stop_search_window_z_end = (
                stop_search_window_z_end,
                stop_search_window_z_begin,
            )
            print(
                f"## compute_Tk: stop search window start and end arguments in the wrong order (for k={k_wavenumber.k_inv_Mpc:.5g}/Mpc). Now searching in interval: z in ({stop_search_window_z_begin}, {stop_search_window_z_end})"
            )

        max_z = z_init.z
        min_z = z_sample.min.z
        if stop_search_window_z_begin > max_z:
            raise ValueError(
                f"compute_Tk: (for k={k_wavenumber.k_inv_Mpc:.5g}/Mpc) specified 'stop' window starting redshift z={stop_search_window_z_begin:.5g} exceeds initial redshift z_init={max_z:.5g}"
            )
        if stop_search_window_z_end < min_z:
            print(
                f"## compute_Tk: (for k={k_wavenumber.k_inv_Mpc:.5g}/Mpc) specified 'stop' window ending redshift z={stop_search_window_z_end:.5g} is smaller than lowest z-source sample point z={min_z:.5g}. Search will terminate at z={min_z:.5g}."
            )
            stop_search_window_z_end = min_z

        if (
            fabs(stop_search_window_z_begin - stop_search_window_z_end)
            < DEFAULT_FLOAT_PRECISION
        ):
            raise ValueError(
                f"## compute_Tk: (for k={k_wavenumber.k_inv_Mpc:.5g}/Mpc) specified search window has zero extent"
            )

        if (
            fabs(z_init.z - z_sample.min.z) < DEFAULT_ABS_TOLERANCE
            or z_init.store_id == z_sample.min.store_id
        ):
            raise ValueError(
                f"## compute_Gk: (for k={k_wavenumber.k_inv_Mpc:.5g}/Mpc) in 'stop' mode, the initial redshift and the lowest source redshift cannot be equal"
            )

    # obtain dimensionful value of wavenumber; this should be measured in the same units used by the cosmology
    # (see below)
    k_float = k_wavenumber.k
    z_min = float(z_sample.min)

    # RHS of ODE system for computing transfer function T_k(z) for gravitational potential Phi(z)
    # We normalize to 1.0 at high redshift, so the initial condition needs to be Phi* (the primordial
    # value of the Newtonian potential) rather than \zeta* (the primordial value of the curvature
    # perturbation on uniform energy hypersurfaces)

    def RHS(z, state, supervisor: TkIntegrationSupervisor) -> List[float]:
        """
        k *must* be measured using the same units used for H(z) in the cosmology
        """
        with RHS_timer(supervisor) as timer:
            T = state[T_INDEX]
            Tprime = state[TPRIME_INDEX]

            if supervisor.notify_available:
                supervisor.message(
                    z,
                    f"current state: T(k) = {T:.5g}, dT(k)/dz = {Tprime:.5g}",
                )
                supervisor.reset_notify_time()

            H = model.functions.Hubble(z)
            wPerturbations = model.functions.wPerturbations(z)
            eps = model.functions.epsilon(z)

            one_plus_z = 1.0 + z
            one_plus_z_2 = one_plus_z * one_plus_z

            dT_dz = Tprime

            k_over_H = k_float / H
            k_over_H_2 = k_over_H * k_over_H

            dTprime_dz = (
                -(eps - 3.0 * (1.0 + wPerturbations)) * Tprime / one_plus_z
                - (
                    (3.0 * (1.0 + wPerturbations) - 2.0 * eps) / one_plus_z_2
                    + wPerturbations * k_over_H_2
                )
                * T
            )

            omega_WKB_sq = Tk_omegaEff_sq(model, k_float, z)

            # try to detect how many oscillations will fit into the log-z grid
            # spacing
            # If the grid spacing is smaller than the oscillation wavelength, then
            # evidently we cannot resolve the oscillations
            if omega_WKB_sq > 0.0:
                wavelength = 2.0 * pi / sqrt(omega_WKB_sq)
                supervisor.report_wavelength(z, wavelength, log((1.0 + z) * k_over_H))

        return [dT_dz, dTprime_dz]

    with TkIntegrationSupervisor(
        k_wavenumber, z_init, z_sample.min, delta_logz=delta_logz
    ) as supervisor:
        initial_state = [1.0, 0.0]

        if mode == "stop":
            # set up an event to terminate the integration when a specified number of e-folds inside the horizon
            def stop_event(z, state, supervisor):
                return z - stop_search_window_z_end + DEFAULT_FLOAT_PRECISION

            # mark stop_event as terminal
            stop_event.terminal = True

            events = [stop_event]
            dense_output = True
        else:
            events = None
            dense_output = False

        sol = solve_ivp(
            RHS,
            method="DOP853",
            t_span=(z_init.z, z_min),
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
            f'compute_Tk: integration did not terminate successfully (k={k_wavenumber.k_inv_Mpc}/Mpc, z_init={z_init.z}, error at z={sol.t[-1]}, "{sol.message}")'
        )

    if mode == "stop" and sol.status != 1:
        raise RuntimeError(
            f'compute_Tk: mode is "{mode}", but integration did not finish following a termination event'
        )

    sampled_z = sol.t
    sampled_values = sol.y
    if len(sampled_z) > 0 and len(sampled_values) != EXPECTED_SOL_LENGTH:
        raise RuntimeError(
            f"compute_Tk: solution does not have expected number of members (expected {EXPECTED_SOL_LENGTH}, found {len(sampled_values)}; k={k_wavenumber.k_inv_Mpc}/Mpc, length of sol.t={len(sampled_z)})"
        )
    if len(sampled_values) > 0:
        sampled_T = sampled_values[T_INDEX]
        sampled_Tprime = sampled_values[TPRIME_INDEX]
    else:
        sampled_T = []
        sampled_Tprime = []

    returned_values = sampled_z.size
    if mode != "stop":
        expected_values = len(z_sample)

        if returned_values != expected_values:
            raise RuntimeError(
                f"compute_Tk: solve_ivp returned {returned_values} samples, but expected {expected_values}"
            )

    stop_deltaz_subh = None
    stop_T = None
    stop_Tprime = None

    if mode == "stop":
        payload = find_phase_minimum(
            sol.sol,
            start_z=stop_search_window_z_begin,
            stop_z=stop_search_window_z_end,
            value_index=T_INDEX,
            deriv_index=TPRIME_INDEX,
        )
        stop_deltaz_subh = k.z_exit - payload["z"]
        stop_T = payload["value"]
        stop_Tprime = payload["derivative"]

    # validate that the samples of the solution correspond to the z-sample points that we specified.
    # This really should be true, but there is no harm in being defensive.
    for i in range(returned_values):
        diff = sampled_z[i] - z_sample[i].z
        if fabs(diff) > DEFAULT_ABS_TOLERANCE:
            raise RuntimeError(
                f"compute_Tk: solve_ivp returned sample points that differ from those requested (difference={diff} at i={i})"
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
        "T_sample": sampled_T,
        "Tprime_sample": sampled_Tprime,
        "solver_label": "solve_ivp+DOP853-stepping0",
        "has_unresolved_osc": supervisor.has_unresolved_osc,
        "unresolved_z": supervisor.unresolved_z,
        "unresolved_efolds_subh": supervisor.unresolved_efolds_subh,
        "stop_deltaz_subh": stop_deltaz_subh,
        "stop_T": stop_T,
        "stop_Tprime": stop_Tprime,
    }


class TkNumericalIntegration(DatastoreObject):
    """
    Encapsulates all sample points produced during a single integration of the
    matter transfer function, labelled by a wavenumber k, and sampled over
    a range of redshifts
    """

    def __init__(
        self,
        payload,
        solver_labels: dict,
        model: ModelProxy,
        k: wavenumber_exit_time,
        atol: tolerance,
        rtol: tolerance,
        z_sample: Optional[redshift_array] = None,
        z_init: Optional[redshift] = None,
        label: Optional[str] = None,
        tags: Optional[List[store_tag]] = None,
        delta_logz: Optional[float] = None,
        mode: Optional[str] = None,
    ):
        k_wavenumber: wavenumber = k.k
        check_units(k_wavenumber, model)

        self._solver_labels = solver_labels
        self._delta_logz = delta_logz
        self._mode = mode.lower() if mode is not None else None

        if self._mode is not None and self._mode not in ["stop"]:
            raise ValueError(
                f'TkNumericalIntegration: unknown compute mode "{self._mode}"'
            )

        self._stop_search_window_start_attr = "z_exit_subh_e3"
        self._stop_search_window_end_attr = "z_exit_subh_e6"

        # if initial time is not really compatible with the initial conditions we use, warn the user
        if z_init is not None and z_init.z < k.z_exit_suph_e3 - DEFAULT_FLOAT_PRECISION:
            print(
                f"!! Warning (TkNumericalIntegration) k={k_wavenumber.k_inv_Mpc:.5g}/Mpc, log10_atol={atol.log10_tol}, log10_rtol={rtol.log10_tol}"
            )
            print(
                f"|    Initial redshift z_init={z_init.z:.5g} is later than the 3-efold superhorizon time z_e3={k.z_exit_suph_e3:.5g}."
            )
            print(
                f"|    Setting initial conditions at this time may lead to meaningless results, because the initial values T_k(z) = 1, Tprime_k(z) = 0"
            )
            print(
                f"|    used for the matter transfer function integration apply only on sufficiently superhorizon scales."
            )

        self._z_sample = z_sample
        if payload is None:
            DatastoreObject.__init__(self, None)
            self._data = None

            self._has_unresolved_osc = None
            self._unresolved_z = None
            self._unresolved_efolds_subh = None

            self._init_efolds_suph = None
            self._stop_deltaz_subh = None
            self._stop_T = None
            self._stop_Tprime = None

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
            self._stop_T = payload["stop_T"]
            self._stop_Tprime = payload["stop_Tprime"]

            self._solver = payload["solver"]

            self._values = payload["values"]

        # check that all sample points are *later* than the specified initial redshift
        if z_init is not None and self._z_sample is not None:
            z_init_float = float(z_init)
            for z in self._z_sample:
                z_float = float(z)
                if z_float > z_init_float:
                    raise ValueError(
                        f"Redshift sample point z={z_float} exceeds initial redshift z={z_init_float}"
                    )

        # store parameters
        self._label = label
        self._tags = tags if tags is not None else []

        self._model_proxy = model

        self._k_exit = k
        self._z_init = z_init

        self._compute_ref = None

        self._atol = atol
        self._rtol = rtol

    @property
    def model_proxy(self) -> ModelProxy:
        return self._model_proxy

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
    def z_init(self) -> redshift:
        return self._z_init

    @property
    def z_sample(self) -> redshift_array:
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
    def stop_T(self) -> float:
        if self._stop_T is None:
            raise RuntimeError("stop_T has not yet been populated")

        return self._stop_T

    @property
    def stop_Tprime(self) -> float:
        if self._stop_Tprime is None:
            raise RuntimeError("stop_Tprime has not yet been populated")

        return self._stop_Tprime

    @property
    def solver(self) -> IntegrationSolver:
        if self._solver is None:
            raise RuntimeError("solver has not yet been populated")
        return self._solver

    @property
    def values(self) -> List:
        if hasattr(self, "_do_not_populate"):
            raise RuntimeError(
                "TkNumericalIntegration: values read but _do_not_populate is set"
            )

        if self._values is None:
            raise RuntimeError("values has not yet been populated")
        return self._values

    def __len__(self):
        if self._values is None:
            return 0

        return len(self._values)

    def __getitem__(self, idx):
        if self._values is None:
            return None

        return self._values[idx]

    def compute(self, label: Optional[str] = None):
        if hasattr(self, "_do_not_populate"):
            raise RuntimeError(
                "TkNumericalIntegration: compute() called but _do_not_populate is set"
            )

        if self._values is not None:
            raise RuntimeError("values have already been computed")

        if self._z_init is None or self._z_sample is None:
            raise RuntimeError(
                "Object has not been configured correctly for a concrete calculation (z_init or z_sample is missing). It can only represent a query."
            )

        # replace label if specified
        if label is not None:
            self._label = label

        # want model to go out of scope, so that it is deleted; we don't want to keep a copy in self,
        # since that would mean it is serialized
        model: BackgroundModel = self._model_proxy.get()

        Hinit = model.functions.Hubble(self.z_init.z)
        k_over_aH = (1.0 + self.z_init.z) * self.k.k / Hinit
        wavelength = 2.0 * pi / k_over_aH
        efolds_suph = -log(k_over_aH)
        if efolds_suph < 1:
            print(
                "!! WARNING (TkNumericalIntegration): T(k) COMPUTATION BEGINNING TOO CLOSE TO HORIZON SCALE"
            )
            print(
                f"|    k = {self.k.k_inv_Mpc}/Mpc, z_exit = {self.z_exit}, z_init = {self.z_init.z}, z_sample(max) = {self.z_sample.max.z}, z_sample(min) = {self.z_sample.min.z}"
            )
            print(
                f"|    k/aH = {k_over_aH:.5g}, wavelength 2pi(H/k) = {wavelength:.5g}, e-folds outside horizon = {efolds_suph}, log(z_init/z_exit) = {log(self.z_init.z/self.z_exit)}"
            )

        # set up limits for the search window used to obtain an initial condition for a subsequent WKB
        # computation of the transfer function.
        # this is done by always cutting at a point of fixed phase where T' = 0 at a minium, so we need to search
        # for such a point, and that search should be performed within a fixed window.
        payload = {}
        if self._mode in ["stop"]:
            payload["mode"] = self._mode

            search_begin = getattr(self._k_exit, self._stop_search_window_start_attr)
            search_end = getattr(self._k_exit, self._stop_search_window_end_attr)

            if search_begin > self._z_init.z:
                search_begin = self._z_init.z

            if search_begin < search_end:
                raise RuntimeError(
                    f"Search window in incorrect order (search_begin={search_begin:.5g}, search_end={search_end:.5g})"
                )

            payload["stop_search_window_z_begin"] = search_begin
            payload["stop_search_window_z_end"] = search_end

        self._compute_ref = compute_Tk.remote(
            self._model_proxy,
            self._k_exit,
            self._z_sample,
            self._z_init,
            atol=self._atol.tol,
            rtol=self._rtol.tol,
            delta_logz=self._delta_logz,
            **payload,
        )
        return self._compute_ref

    def store(self) -> Optional[bool]:
        if self._compute_ref is None:
            raise RuntimeError(
                "TkNumericalIntegration: store() called, but no compute() is in progress"
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
        self._stop_T = data.get("stop_T", None)
        self._stop_Tprime = data.get("stop_Tprime", None)

        model: BackgroundModel = self._model_proxy.get()

        Hinit = model.functions.Hubble(self.z_init.z)
        k_over_aH = (1.0 + self.z_init.z) * self.k.k / Hinit
        self._init_efolds_suph = -log(k_over_aH)

        T_sample = data["T_sample"]
        Tprime_sample = data["Tprime_sample"]
        self._values = []

        for i in range(len(T_sample)):
            current_z = self._z_sample[i]
            current_z_float = current_z.z
            H = model.functions.Hubble(current_z_float)
            tau = model.functions.tau(current_z_float)
            wPerturbations = model.functions.wPerturbations(current_z_float)

            analytic_T_rad = compute_analytic_T(self.k.k, 1.0 / 3.0, tau)
            analytic_Tprime_rad = compute_analytic_Tprime(self.k.k, 1.0 / 3.0, tau, H)

            analytic_T_w = compute_analytic_T(self.k.k, wPerturbations, tau)
            analytic_Tprime_w = compute_analytic_Tprime(
                self.k.k, wPerturbations, tau, H
            )

            omega_WKB_sq = Tk_omegaEff_sq(model, self.k.k, current_z_float)
            WKB_criterion = fabs(
                Tk_d_ln_omegaEffPrime_dz(model, self.k.k, current_z_float)
            ) / sqrt(fabs(omega_WKB_sq))

            # create new TkNumericalValue object
            self._values.append(
                TkNumericalValue(
                    None,
                    current_z,
                    T_sample[i],
                    Tprime_sample[i],
                    analytic_T_rad=analytic_T_rad,
                    analytic_Tprime_rad=analytic_Tprime_rad,
                    analytic_T_w=analytic_T_w,
                    analytic_Tprime_w=analytic_Tprime_w,
                    omega_WKB_sq=omega_WKB_sq,
                    WKB_criterion=WKB_criterion,
                )
            )

        self._solver = self._solver_labels[data["solver_label"]]

        return True


class TkNumericalValue(DatastoreObject):
    """
    Encapsulates a single sampled value of the matter transfer functions.
    Parameters such as wavenumber k, initial redshift z_init, etc., are held by the
    owning TkNumericalIntegration object
    """

    def __init__(
        self,
        store_id: int,
        z: redshift,
        T: float,
        Tprime: float,
        analytic_T_rad: Optional[float] = None,
        analytic_Tprime_rad: Optional[float] = None,
        analytic_T_w: Optional[float] = None,
        analytic_Tprime_w: Optional[float] = None,
        omega_WKB_sq: Optional[float] = None,
        WKB_criterion: Optional[float] = None,
    ):
        DatastoreObject.__init__(self, store_id)

        self._z = z
        self._T = T
        self._Tprime = Tprime

        self._analytic_T_rad = analytic_T_rad
        self._analytic_Tprime_rad = analytic_Tprime_rad

        self._analytic_T_w = analytic_T_w
        self._analytic_Tprime_w = analytic_Tprime_w

        self._omega_WKB_sq = omega_WKB_sq
        self._WKB_criterion = WKB_criterion

    def __float__(self):
        """
        Cast to float. Returns value of the transfer function
        :return:
        """
        return self.T

    @property
    def z(self) -> redshift:
        return self._z

    @property
    def T(self) -> float:
        return self._T

    @property
    def Tprime(self) -> float:
        return self._Tprime

    @property
    def analytic_T_rad(self) -> Optional[float]:
        return self._analytic_T_rad

    @property
    def analytic_Tprime_rad(self) -> Optional[float]:
        return self._analytic_Tprime_rad

    @property
    def analytic_T_w(self) -> Optional[float]:
        return self._analytic_T_w

    @property
    def analytic_Tprime_w(self) -> Optional[float]:
        return self._analytic_Tprime_w

    @property
    def omega_WKB_sq(self) -> Optional[float]:
        return self._omega_WKB_sq

    @property
    def WKB_criterion(self) -> Optional[float]:
        return self._WKB_criterion
