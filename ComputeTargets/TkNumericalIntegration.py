from math import fabs, pi, log, sqrt
from typing import Optional, List

import ray

from ComputeTargets.BackgroundModel import BackgroundModel, ModelProxy
from ComputeTargets.WKB_Tk import Tk_omegaEff_sq, Tk_d_ln_omegaEff_dz
from ComputeTargets.analytic_Tk import compute_analytic_T, compute_analytic_Tprime
from CosmologyConcepts import redshift_array, wavenumber, redshift, wavenumber_exit_time
from Datastore import DatastoreObject
from MetadataConcepts import tolerance, store_tag
from Quadrature.integration_metadata import IntegrationSolver, IntegrationData
from Quadrature.integrators.numerical_with_phase_cut import (
    VALUE_INDEX,
    DERIV_INDEX,
    numerical_with_phase_cut,
)
from Quadrature.supervisors.base import (
    RHS_timer,
)
from Quadrature.supervisors.numerical import NumericalIntegrationSupervisor
from Units import check_units
from defaults import (
    DEFAULT_FLOAT_PRECISION,
)


# RHS of ODE system
#
# State layout:
#   state[0] = T(z)
#   state[1] = dT/dz = T' = "T prime

# RHS of ODE system for computing transfer function T_k(z) for gravitational potential Phi(z)
# We normalize to 1.0 at high redshift, so the initial condition needs to be Phi* (the primordial
# value of the Newtonian potential) rather than \zeta* (the primordial value of the curvature
# perturbation on uniform energy hypersurfaces)


def RHS(
    z: float,
    state: List[float],
    model: BackgroundModel,
    k_float: float,
    supervisor: NumericalIntegrationSupervisor,
) -> List[float]:
    """
    k *must* be measured using the same units used for H(z) in the cosmology
    """
    with RHS_timer(supervisor) as timer:
        T = state[VALUE_INDEX]
        Tprime = state[DERIV_INDEX]

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
        z_init: Optional[redshift] = None,
        z_sample: Optional[redshift_array] = None,
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

        # search for a handover point (to the WKB calculation) from 3 to 6 e-folds inside the horizon
        self._stop_search_window_start_attr = "z_exit_subh_e3"
        self._stop_search_window_end_attr = "z_exit_subh_e6"

        # if initial time is not really compatible with the initial conditions we use, warn the user
        if z_init is not None and z_init.z < k.z_exit_suph_e3 - DEFAULT_FLOAT_PRECISION:
            print(
                f"!! Warning (TkNumericalIntegration) k={k_wavenumber.k_inv_Mpc:.5g}/Mpc, log10_atol={atol.log10_tol}, log10_rtol={rtol.log10_tol}\n"
                f"|    Initial redshift z_init={z_init.z:.5g} is later than the 3-efold superhorizon time z_e3={k.z_exit_suph_e3:.5g}.\n"
                f"|    Setting initial conditions at this time may lead to meaningless results, because the initial values T_k(z) = 1, Tprime_k(z) = 0.\n"
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
                        f"Redshift sample point z={z_float} is earlier than initial redshift z={z_init_float}"
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

        self._compute_ref = numerical_with_phase_cut.remote(
            self._model_proxy,
            self._k_exit,
            self._z_init,
            self._z_sample,
            initial_value=1.0,
            initial_deriv=0.0,
            RHS=RHS,
            atol=self._atol.tol,
            rtol=self._rtol.tol,
            delta_logz=self._delta_logz,
            task_label="compute_Tk",
            object_label="Tk(z)",
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
        self._stop_T = data.get("stop_value", None)
        self._stop_Tprime = data.get("stop_deriv", None)

        model: BackgroundModel = self._model_proxy.get()

        Hinit = model.functions.Hubble(self.z_init.z)
        k_over_aH = (1.0 + self.z_init.z) * self.k.k / Hinit
        self._init_efolds_suph = -log(k_over_aH)

        T_sample = data["value_sample"]
        Tprime_sample = data["deriv_sample"]

        # need to be aware that T_sample may not be as long as self._z_sample, if we are working in "stop" mode
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

            omega_sq = Tk_omegaEff_sq(model, self.k.k, current_z_float)
            WKB_criterion = fabs(
                Tk_d_ln_omegaEff_dz(model, self.k.k, current_z_float)
            ) / sqrt(fabs(omega_sq))

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
                    omega_WKB_sq=omega_sq,
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
