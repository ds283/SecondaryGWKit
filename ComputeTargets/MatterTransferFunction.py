from math import fabs
from typing import Optional, List
import time
from traceback import print_tb

import ray
from scipy.integrate import solve_ivp

from CosmologyConcepts import redshift_array, wavenumber, redshift, wavenumber_exit_time
from CosmologyModels import BaseCosmology
from Datastore import DatastoreObject
from MetadataConcepts import tolerance, store_tag
from defaults import DEFAULT_ABS_TOLERANCE, DEFAULT_REL_TOLERANCE
from utilities import check_units, WallclockTimer, format_time
from .integration_metadata import IntegrationSolver


DEFAULT_UPDATE_INTERVAL = 1*60


class IntegrationSupervisor:
    def __init__(self, notify_interval: int=DEFAULT_UPDATE_INTERVAL):
        self._notify_interval: int = notify_interval

        self._RHS_time: float = 0
        self._RHS_evaluations: int = 0

        self._min_RHS_time: float = None
        self._max_RHS_time: float = None

        self._num_notifications = 0

    def __enter__(self):
        self._start_time = time.time()
        self._last_notify = self._start_time

        self._integration_start = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._integration_end = time.perf_counter()
        self.integration_time = self._integration_end - self._integration_start

        if exc_type is not None:
            print(f"type={exc_type}, value={exc_val}")
            print_tb(exc_tb)

    @property
    def notify_available(self) -> bool:
        return time.time() - self._last_notify > self._notify_interval

    def report_notify(self) -> int:
        self._num_notifications += 1
        return self._num_notifications

    def reset_notify_time(self):
        self._last_notify = time.time()

    def notify_new_RHS_time(self, RHS_time):
        self._RHS_time = self._RHS_time + RHS_time
        self._RHS_evaluations += 1

        if self._min_RHS_time is None or RHS_time < self._min_RHS_time:
            self._min_RHS_time = RHS_time

        if self._max_RHS_time is None or RHS_time > self._max_RHS_time:
            self._max_RHS_time = RHS_time

    @property
    def mean_RHS_time(self) -> float:
        if self._RHS_evaluations == 0:
            return None

        return self._RHS_time/self._RHS_evaluations

    @property
    def min_RHS_time(self) -> float:
        return self._min_RHS_time

    @property
    def max_RHS_time(self) -> float:
        return self._max_RHS_time

    @property
    def RHS_evaluations(self) -> int:
        return self._RHS_evaluations


class MatterTransferFunctionSupervisor(IntegrationSupervisor):
    def __init__(self, k: wavenumber, z_init: redshift, z_final: redshift, notify_interval: int=DEFAULT_UPDATE_INTERVAL):
        super().__init__(notify_interval)

        self._k: wavenumber = k
        self._z_init: float = z_init.z
        self._z_final: float = z_final.z

        self._z_range: float = self._z_init - self._z_final

        self._last_z: float = self._z_init

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
        print(f"** STATUS UPDATE #{update_number}: Integration for T(k) for k = {self._k.k_inv_Mpc:.5g}/Mpc (store_id={self._k.store_id}) has been running for {format_time(since_start)} ({format_time(since_last_notify)} since last notification)")
        print(f"|    current z={current_z:.5g} (initial z={self._z_init:.5g}, target z={self._z_final:.5g}, z complete={z_complete:.5g}, z remain={z_remain:.5g}, {percent_remain:.3g}% remains)")
        if self._last_z is not None:
            z_delta = self._last_z - current_z
            print(f"|    redshift advance since last update: Delta z = {z_delta:.5g}")
        print(f"|    {self.RHS_evaluations} RHS evaluations, mean {self.mean_RHS_time:.5g}s per evaluation, min RHS time = {self.min_RHS_time:.5g}s, max RHS time = {self.max_RHS_time:.5g}s")
        print(f"|    {msg}")

        self._last_z = current_z

class _RHS_timer:
    def __init__(self, supervisor: IntegrationSupervisor):
        self._supervisor = supervisor

    def __enter__(self):
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end_time = time.perf_counter()
        self._elapsed = self._end_time - self._start_time

        self._supervisor.notify_new_RHS_time(self._elapsed)

        if exc_type is not None:
            print(f"type={exc_type}, value={exc_val}")
            print_tb(exc_tb)


@ray.remote
def compute_matter_Tk(
    cosmology: BaseCosmology,
    k: wavenumber,
    z_sample: redshift_array,
    z_init: redshift,
    atol: float = DEFAULT_ABS_TOLERANCE,
    rtol: float = DEFAULT_REL_TOLERANCE,
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
    #   state[0] = T(z)
    #   state[1] = dT/dz = T' = "T prime"
    def RHS(z, state, supervisor: MatterTransferFunctionSupervisor) -> float:
        """
        k *must* be measured using the same units used for H(z) in the cosmology
        """
        with _RHS_timer(supervisor):
            rho, T, Tprime = state

            if supervisor.notify_available:
                supervisor.message(z, f"current state: rho = {rho:.5g}, T(k) = {T:.5g}, dT(k)/dz = {Tprime:.5g}")
                supervisor.reset_notify_time()

            H = cosmology.Hubble(z)
            wBackground = cosmology.wBackground(z)
            wPerturbations = cosmology.wPerturbations(z)
            eps = cosmology.epsilon(z)

            one_plus_z = 1.0 + z
            one_plus_z_2 = one_plus_z * one_plus_z

            drho_dz = 3.0 * ((1.0 + wBackground) / one_plus_z) * rho
            dT_dz = Tprime

            k_over_H = k_float / H
            k_over_H_2 = k_over_H * k_over_H

            dTprime_dz = (
                -(eps - 3.0 * (1.0 + wPerturbations)) * Tprime / one_plus_z
                - (3.0 * (1.0 + wPerturbations) - 2.0 * eps) * T / one_plus_z_2
                - wPerturbations * k_over_H_2 * T
            )

        return [drho_dz, dT_dz, dTprime_dz]

    with MatterTransferFunctionSupervisor(k, z_init, z_sample.min) as supervisor:
        # use initial values T(z) = 1, dT/dz = 0 at z = z_init
        initial_state = [cosmology.rho(z_init.z), 1.0, 0.0]
        sol = solve_ivp(
            RHS,
            method="RK45",
            t_span=(z_init.z, z_min),
            y0=initial_state,
            t_eval=z_sample.as_list(),
            atol=atol,
            rtol=rtol,
            args=(supervisor,)
        )

    # test whether the integration concluded successfully
    if not sol.success:
        raise RuntimeError(
            f'compute_matter_Tk: integration did not terminate successfully (k={k.k_inv_Mpc}/Mpc, z_init={z_init.z}, error at z={sol.t[-1]}, "{sol.message}")'
        )

    sampled_z = sol.t
    sampled_values = sol.y
    sampled_T = sampled_values[1]

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
        "values": sampled_T,
        "solver_label": "scipy+solve_ivp+RK45",
        "solver_stepping": 0,
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
        cosmology: BaseCosmology,
        k: wavenumber_exit_time,
        z_sample: redshift_array,
        z_init: redshift,
        atol: tolerance,
        rtol: tolerance,
        label: Optional[str] = None,
        tags: Optional[List[store_tag]] = None,
    ):
        check_units(k.k, cosmology)

        if z_init.z < 10.0 * k.z_exit:
            raise RuntimeError(
                f"Initial time z_init={z_init.z:.5g} is not sufficiently before horizon exit for k={k.k.k_inv_Mpc:.5g}/Mpc (z_exit={k.z_exit:.5g})"
            )

        if payload is None:
            DatastoreObject.__init__(self, None)
            self._compute_time = None
            self._compute_steps = None
            self._solver = None

            self._z_sample = z_sample
            self._values = None
        else:
            DatastoreObject.__init__(self, payload["store_id"])
            self._compute_time = payload["compute_time"]
            self._compute_steps = payload["compute_steps"]
            self._solver = payload["solver"]

            self._z_sample = z_sample
            self._values = payload["values"]

        # check that all sample points are *later* than the specified initial redshift
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
    def solver(self) -> float:
        if self._solver is None:
            raise RuntimeError("compute_steps has not yet been populated")
        return self._solver

    @property
    def values(self) -> List:
        if self._values is None:
            raise RuntimeError("values has not yet been populated")
        return self._values

    def compute(self, label: Optional[str] = None):
        if self._values is not None:
            raise RuntimeError("values have already been computed")

        # replace label if specified
        if label is not None:
            self._label = label

        # print("@@ BEGINNING T(k) COMPUTATION")
        # print(
        #     f"     k = {self.k.k_inv_Mpc}/Mpc, z_exit = {self.z_exit}, z_init = {self.z_init.z}, z_sample(max) = {self.z_sample.max.z}, z_sample(min) = {self.z_sample.min.z}"
        # )

        self._compute_ref = compute_matter_Tk.remote(
            self.cosmology,
            self.k,
            self.z_sample,
            self.z_init,
            atol=self._atol.tol,
            rtol=self._rtol.tol,
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

        values = data["values"]
        self._values = []

        for i in range(len(values)):
            # create new MatterTransferFunctionValue object
            self._values.append(
                MatterTransferFunctionValue(None, self._z_sample[i], values[i])
            )

        self._solver = IntegrationSolver(
            store_id=None, label=data["solver_label"], stepping=data["solver_stepping"]
        )

        return True


class MatterTransferFunctionValue(DatastoreObject):
    """
    Encapsulates a single sampled value of the matter transfer functions.
    Parameters such as wavenumber k, intiial redshift z_init, etc., are held by the
    owning MatterTransferFunctionIntegration object
    """

    def __init__(self, store_id: int, z: redshift, value: float):
        DatastoreObject.__init__(self, store_id)

        self._z = z
        self._value = value

    def __float__(self):
        """
        Cast to float. Returns value of the transfer function
        :return:
        """
        return self.value

    @property
    def z(self) -> redshift:
        return self._z

    @property
    def value(self) -> float:
        return self._value
