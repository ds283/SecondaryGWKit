import time
from typing import Optional, List

import ray
from math import fabs, pi, log, sqrt, pow
from scipy.integrate import solve_ivp
from scipy.special import gamma, jv

from CosmologyConcepts import redshift_array, wavenumber, redshift, wavenumber_exit_time
from CosmologyModels import BaseCosmology
from Datastore import DatastoreObject
from MetadataConcepts import tolerance, store_tag
from defaults import DEFAULT_ABS_TOLERANCE, DEFAULT_REL_TOLERANCE
from utilities import check_units, format_time
from .integration_metadata import IntegrationSolver
from .integration_supervisor import (
    DEFAULT_UPDATE_INTERVAL,
    IntegrationSupervisor,
    RHS_timer,
)


def analytic_matter_transfer_function(k, w, tau):
    b = (1.0 - 3.0 * w) / (1.0 + 3.0 * w)
    cs = sqrt(w)
    k_cs_tau = k * cs * tau

    A = pow(2.0, 1.5 + b)
    B = gamma(2.5 + b)
    C = pow(k_cs_tau, -1.5 - b)
    D = jv(1.5 + b, k_cs_tau)

    return A * B * C * D


class MatterTransferFunctionSupervisor(IntegrationSupervisor):
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
        percent_remain = 100.0 * (z_remain / self._z_range)
        print(
            f"** STATUS UPDATE #{update_number}: Integration for T(k) for k = {self._k.k_inv_Mpc:.5g}/Mpc (store_id={self._k.store_id}) has been running for {format_time(since_start)} ({format_time(since_last_notify)} since last notification)"
        )
        print(
            f"|    current z={current_z:.5g} (initial z={self._z_init:.5g}, target z={self._z_final:.5g}, z complete={z_complete:.5g}, z remain={z_remain:.5g}, {percent_remain:.3g}% remains)"
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
def compute_matter_Tk(
    cosmology: BaseCosmology,
    k: wavenumber,
    z_sample: redshift_array,
    z_init: redshift,
    atol: float = DEFAULT_ABS_TOLERANCE,
    rtol: float = DEFAULT_REL_TOLERANCE,
    delta_logz: Optional[float] = None,
) -> dict:
    check_units(k, cosmology)

    # obtain dimensionful value of wavenumber; this should be measured in the same units used by the cosmology
    # (see below)
    k_float = k.k
    z_min = float(z_sample.min)

    # RHS of ODE system for computing transfer function T_k(z) for gravitational potential Phi(z)
    # We normalize to 1.0 at high redshift, so the initial condition needs to be Phi* (the primordial
    # value of the Newtonian potential) rather than \zeta* (the primordial value of the curvature
    # perturbation on uniform energy hypersurfaces)
    #
    # State layout:
    #   state[0] = energy density rho(z)
    #   state[1] = a_0 tau(z) [tau = conformal time]
    #   state[2] = T(z)
    #   state[3] = dT/dz = T' = "T prime"
    RHO_INDEX = 0
    A0_TAU_INDEX = 1
    T_INDEX = 2
    TPRIME_INDEX = 3

    def RHS(z, state, supervisor: MatterTransferFunctionSupervisor) -> float:
        """
        k *must* be measured using the same units used for H(z) in the cosmology
        """
        with RHS_timer(supervisor):
            rho = state[RHO_INDEX]
            a0_tau = state[A0_TAU_INDEX]
            T = state[T_INDEX]
            Tprime = state[TPRIME_INDEX]

            if supervisor.notify_available:
                supervisor.message(
                    z,
                    f"current state: rho = {rho:.5g}, T(k) = {T:.5g}, dT(k)/dz = {Tprime:.5g}",
                )
                supervisor.reset_notify_time()

            H = cosmology.Hubble(z)
            wBackground = cosmology.wBackground(z)
            wPerturbations = cosmology.wPerturbations(z)
            eps = cosmology.epsilon(z)

            one_plus_z = 1.0 + z
            one_plus_z_2 = one_plus_z * one_plus_z

            drho_dz = 3.0 * ((1.0 + wBackground) / one_plus_z) * rho
            da0_tau_dz = -1.0 / H
            dT_dz = Tprime

            k_over_H = k_float / H
            k_over_H_2 = k_over_H * k_over_H

            omega_eff_sq = (
                3.0 * (1.0 + wPerturbations) - 2.0 * eps
            ) / one_plus_z_2 + wPerturbations * k_over_H_2

            dTprime_dz = (
                -(eps - 3.0 * (1.0 + wPerturbations)) * Tprime / one_plus_z
                - omega_eff_sq * T
            )

            # try to detect how many oscillations will fit into the log-z grid
            # spacing
            # If the grid spacing is smaller than the oscillation wavelength, then
            # evidently we cannot resolve the oscillations
            if omega_eff_sq > 0.0:
                wavelength = 2.0 * pi / sqrt(omega_eff_sq)
                supervisor.report_wavelength(z, wavelength, log((1.0 + z) * k_over_H))

        return [drho_dz, da0_tau_dz, dT_dz, dTprime_dz]

    with MatterTransferFunctionSupervisor(
        k, z_init, z_sample.min, delta_logz=delta_logz
    ) as supervisor:
        # use initial values T(z) = 1, dT/dz = 0 at z = z_init
        rho_init = cosmology.rho(z_init.z)
        tau_init = (
            sqrt(3.0) * cosmology.units.PlanckMass / sqrt(rho_init) / (1.0 + z_init.z)
        )

        initial_state = [rho_init, tau_init, 1.0, 0.0]
        sol = solve_ivp(
            RHS,
            method="Radau",
            t_span=(z_init.z, z_min),
            y0=initial_state,
            t_eval=z_sample.as_list(),
            atol=atol,
            rtol=rtol,
            args=(supervisor,),
        )

    # test whether the integration concluded successfully
    if not sol.success:
        raise RuntimeError(
            f'compute_matter_Tk: integration did not terminate successfully (k={k.k_inv_Mpc}/Mpc, z_init={z_init.z}, error at z={sol.t[-1]}, "{sol.message}")'
        )

    sampled_z = sol.t
    sampled_values = sol.y
    sampled_a0_tau = sampled_values[A0_TAU_INDEX]
    sampled_T = sampled_values[T_INDEX]
    sampled_Tprime = sampled_values[TPRIME_INDEX]

    expected_values = len(z_sample)
    returned_values = sampled_z.size

    if returned_values != expected_values:
        raise RuntimeError(
            f"compute_matter_Tk: solve_ivp returned {returned_values} samples, but expected {expected_values}"
        )

    for i in range(returned_values):
        diff = sampled_z[i] - z_sample[i].z
        if fabs(diff) > DEFAULT_ABS_TOLERANCE:
            raise RuntimeError(
                f"compute_matter_Tk: solve_ivp returned sample points that differ from those requested (difference={diff} at i={i})"
            )

    return {
        "compute_time": supervisor.integration_time,
        "compute_steps": int(sol.nfev),
        "RHS_evaluations": supervisor.RHS_evaluations,
        "mean_RHS_time": supervisor.mean_RHS_time,
        "max_RHS_time": supervisor.max_RHS_time,
        "min_RHS_time": supervisor.min_RHS_time,
        "a0_tau_sample": sampled_a0_tau,
        "T_sample": sampled_T,
        "Tprime_sample": sampled_Tprime,
        "solver_label": "solve_ivp+Radau-stepping0",
        "has_unresolved_osc": supervisor.has_unresolved_osc,
        "unresolved_z": supervisor.unresolved_z,
        "unresolved_efolds_subh": supervisor.unresolved_efolds_subh,
    }


class MatterTransferFunctionIntegration(DatastoreObject):
    """
    Encapsulates all sample points produced during a single integration of the
    matter transfer function, labelled by a wavenumber k, and sampled over
    a range of redshifts
    """

    def __init__(
        self,
        payload,
        solver_labels: dict,
        cosmology: BaseCosmology,
        k: wavenumber_exit_time,
        atol: tolerance,
        rtol: tolerance,
        z_sample: Optional[redshift_array] = None,
        z_init: Optional[redshift] = None,
        label: Optional[str] = None,
        tags: Optional[List[store_tag]] = None,
        delta_logz: Optional[float] = None,
    ):
        check_units(k.k, cosmology)

        self._solver_labels = solver_labels
        self._delta_logz = delta_logz

        if z_init.z < 10.0 * k.z_exit:
            raise RuntimeError(
                f"Initial time z_init={z_init.z:.5g} is not sufficiently before horizon exit for k={k.k.k_inv_Mpc:.5g}/Mpc (z_exit={k.z_exit:.5g})"
            )

        if payload is None:
            DatastoreObject.__init__(self, None)
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

            self._solver = None

            self._z_sample = z_sample
            self._values = None
        else:
            DatastoreObject.__init__(self, payload["store_id"])
            self._compute_time = payload["compute_time"]
            self._compute_steps = payload["compute_steps"]
            self._RHS_evaluations = payload["RHS_evaluations"]
            self._mean_RHS_time = payload["mean_RHS_time"]
            self._max_RHS_time = payload["max_RHS_time"]
            self._min_RHS_time = payload["min_RHS_time"]

            self._has_unresolved_osc = payload["has_unresolved_osc"]
            self._unresolved_z = payload["unresolved_z"]
            self._unresolved_efolds_subh = payload["unresolved_efolds_subh"]

            self._init_efolds_suph = payload["init_efolds_suph"]

            self._solver = payload["solver"]

            self._z_sample = z_sample
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

        self._cosmology = cosmology

        self._k_exit = k
        self._z_init = z_init

        self._compute_ref = None

        self._atol = atol
        self._rtol = rtol

    @property
    def cosmology(self) -> BaseCosmology:
        return self._cosmology

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
    def compute_time(self) -> float:
        if self._compute_time is None:
            raise RuntimeError("compute_time has not yet been populated")
        return self._compute_time

    @property
    def compute_steps(self) -> int:
        if self._compute_time is None:
            raise RuntimeError("compute_steps has not yet been populated")
        return self._compute_steps

    @property
    def mean_RHS_time(self) -> int:
        if self._mean_RHS_time is None:
            raise RuntimeError("mean_RHS_time has not yet been populated")
        return self._mean_RHS_time

    @property
    def max_RHS_time(self) -> int:
        if self._max_RHS_time is None:
            raise RuntimeError("max_RHS_time has not yet been populated")
        return self._max_RHS_time

    @property
    def min_RHS_time(self) -> int:
        if self._min_RHS_time is None:
            raise RuntimeError("min_RHS_time has not yet been populated")
        return self._min_RHS_time

    @property
    def RHS_evaluations(self) -> int:
        if self._RHS_evaluations is None:
            raise RuntimeError("RHS_evaluations has not yet been populated")
        return self._RHS_evaluations

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
    def solver(self) -> IntegrationSolver:
        if self._solver is None:
            raise RuntimeError("solver has not yet been populated")
        return self._solver

    @property
    def values(self) -> List:
        if self._values is None:
            raise RuntimeError("values has not yet been populated")
        return self._values

    def compute(self, label: Optional[str] = None):
        if self._values is not None:
            raise RuntimeError("values have already been computed")

        if self._z_init is None or self._z_sample is None:
            raise RuntimeError(
                "Object has not been configured correctly for a concrete calculation (z_init or z_sample is missing). It can only represent a query."
            )

        # replace label if specified
        if label is not None:
            self._label = label

        Hinit = self.cosmology.Hubble(self.z_init.z)
        k_over_aH = (1.0 + self.z_init.z) * self.k.k / Hinit
        wavelength = 2.0 * pi / k_over_aH
        efolds_suph = -log(k_over_aH)
        if efolds_suph < 1:
            print("!! T(k) COMPUTATION BEGINNING TOO CLOSE TO HORIZON SCALE")
            print(
                f"|    k = {self.k.k_inv_Mpc}/Mpc, z_exit = {self.z_exit}, z_init = {self.z_init.z}, z_sample(max) = {self.z_sample.max.z}, z_sample(min) = {self.z_sample.min.z}"
            )
            print(
                f"|    k/aH = {k_over_aH:.5g}, wavelength 2pi(H/k) = {wavelength:.5g}, e-folds outside horizon = {efolds_suph}, log(z_init/z_exit) = {log(self.z_init.z/self.z_exit)}"
            )

        self._compute_ref = compute_matter_Tk.remote(
            self.cosmology,
            self.k,
            self.z_sample,
            self.z_init,
            atol=self._atol.tol,
            rtol=self._rtol.tol,
            delta_logz=self._delta_logz,
        )
        return self._compute_ref

    def store(self) -> Optional[bool]:
        if self._compute_ref is None:
            raise RuntimeError(
                "MatterTransferFunctionIntegration: store() called, but no compute() is in progress"
            )

        # check whether the computation has actually resolved
        resolved, unresolved = ray.wait([self._compute_ref], timeout=0)

        # if not, return None
        if len(resolved) == 0:
            return None

        # retrieve result and populate ourselves
        data = ray.get(self._compute_ref)
        self._compute_ref = None

        self._compute_time = data["compute_time"]
        self._compute_steps = data["compute_steps"]
        self._RHS_evaluations = data["RHS_evaluations"]
        self._mean_RHS_time = data["mean_RHS_time"]
        self._max_RHS_time = data["max_RHS_time"]
        self._min_RHS_time = data["min_RHS_time"]

        self._has_unresolved_osc = data["has_unresolved_osc"]
        self._unresolved_z = data["unresolved_z"]
        self._unresolved_efolds_subh = data["unresolved_efolds_subh"]

        Hinit = self.cosmology.Hubble(self.z_init.z)
        k_over_aH = (1.0 + self.z_init.z) * self.k.k / Hinit
        self._init_efolds_suph = -log(k_over_aH)

        T_sample = data["T_sample"]
        Tprime_sample = data["Tprime_sample"]
        a0_tau_sample = data["a0_tau_sample"]
        self._values = []

        for i in range(len(T_sample)):
            tau = a0_tau_sample[i]
            analytic_T = analytic_matter_transfer_function(self.k.k, 1.0 / 3.0, tau)

            # create new MatterTransferFunctionValue object
            self._values.append(
                MatterTransferFunctionValue(
                    None,
                    self._z_sample[i],
                    T_sample[i],
                    Tprime_sample[i],
                    analytic_T=analytic_T,
                )
            )

        self._solver = self._solver_labels[data["solver_label"]]

        return True


class MatterTransferFunctionValue(DatastoreObject):
    """
    Encapsulates a single sampled value of the matter transfer functions.
    Parameters such as wavenumber k, intiial redshift z_init, etc., are held by the
    owning MatterTransferFunctionIntegration object
    """

    def __init__(
        self,
        store_id: int,
        z: redshift,
        T: float,
        Tprime: float,
        analytic_T: Optional[float] = None,
    ):
        DatastoreObject.__init__(self, store_id)

        self._z = z
        self._T = T
        self._Tprime = Tprime
        self._analytic_T = analytic_T

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
    def analytic_T(self) -> Optional[float]:
        return self._analytic_T
