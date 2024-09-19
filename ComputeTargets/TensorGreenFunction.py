from math import fabs
from typing import Optional, List

import ray
from scipy.integrate import solve_ivp

from CosmologyConcepts import wavenumber, redshift, redshift_array, wavenumber_exit_time
from CosmologyModels import BaseCosmology
from Datastore import DatastoreObject
from MetadataConcepts import tolerance, store_tag
from defaults import DEFAULT_ABS_TOLERANCE, DEFAULT_REL_TOLERANCE
from utilities import check_units, format_time
from .integration_metadata import IntegrationSolver
from .integration_supervisor import (
    IntegrationSupervisor,
    DEFAULT_UPDATE_INTERVAL,
    RHS_timer,
)


class TensorGreenFunctionSupervisor(IntegrationSupervisor):
    def __init__(
        self,
        k: wavenumber,
        z_source: redshift,
        z_final: redshift,
        notify_interval: int = DEFAULT_UPDATE_INTERVAL,
    ):
        super().__init__(notify_interval)

        self._k: wavenumber = k
        self._z_source: float = z_source.z
        self._z_final: float = z_final.z

        self._z_range: float = self._z_source - self._z_final

        self._last_z: float = self._z_source

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


@ray.remote
def compute_tensor_Green(
    cosmology: BaseCosmology,
    k: wavenumber,
    z_source: redshift,
    z_sample: redshift_array,
    atol: float = DEFAULT_ABS_TOLERANCE,
    rtol: float = DEFAULT_REL_TOLERANCE,
) -> dict:
    check_units(k, cosmology)

    # obtain dimensionful value of wavenumber; this should be measured in the same units used by the cosmology
    # (see below)
    k_float = k.k
    z_min = float(z_sample.min)

    Hsource = cosmology.Hubble(z_source.z)

    # RHS of ODE system
    #
    # State layout:
    #   state[0] = energy density rho(z)
    #   state[0] = G_k(z, z')
    #   state[1] = Gprime_k(z, z')
    def RHS(z, state, supervisor) -> float:
        """
        k *must* be measured using the same units used for H(z) in the cosmology
        """
        with RHS_timer(supervisor):
            rho, G, Gprime = state

            if supervisor.notify_available:
                supervisor.message(
                    z,
                    f"current state: rho = {rho:.5g}, Gr(k) = {G:.5g}, dGr(k)/dz = {Gprime:.5g}",
                )
                supervisor.reset_notify_time()

            H = cosmology.Hubble(z)
            w = cosmology.wBackground(z)
            eps = cosmology.epsilon(z)

            one_plus_z = 1.0 + z
            one_plus_z_2 = one_plus_z * one_plus_z

            drho_dz = 3.0 * ((1.0 + w) / one_plus_z) * rho
            dG_dz = Gprime

            k_over_H = k_float / H
            k_over_H_2 = k_over_H * k_over_H

            dGprime_dz = (
                -(eps / one_plus_z) * Gprime
                - (k_over_H_2 + (eps - 2.0) / one_plus_z_2) * G
            )

        return [drho_dz, dG_dz, dGprime_dz]

    with TensorGreenFunctionSupervisor(k, z_source, z_sample.min) as supervisor:
        # initial conditions should be
        #   G(z', z') = 0
        #   Gprime(z' z') = -1/(a0 H(z'))
        # however we would rather not have a delicate initial condition for Gprime, so we
        # instead solve with the boundary conditions Gprime = -1 and rescale afterwards
        initial_state = [cosmology.rho(z_source.z), 0.0, 1.0]
        sol = solve_ivp(
            RHS,
            method="Radau",
            t_span=(z_source.z, z_min),
            y0=initial_state,
            t_eval=z_sample.as_list(),
            atol=atol,
            rtol=rtol,
            args=(supervisor,),
        )

    # test whether the integration concluded successfully
    if not sol.success:
        raise RuntimeError(
            f'compute_tensor_Green: integration did not terminate successfully (k={k.k_inv_Mpc}/Mpc, z_source={z_source.z}, error at z={sol.t[-1]}, "{sol.message}")'
        )

    sampled_z = sol.t
    sampled_values = sol.y
    sampled_G = sampled_values[1] / Hsource

    expected_values = len(z_sample)
    returned_values = sampled_z.size

    if returned_values != expected_values:
        raise RuntimeError(
            f"compute_tensor_Green: solve_ivp returned {returned_values} samples, but expected {expected_values}"
        )

    for i in range(returned_values):
        diff = sampled_z[i] - z_sample[i].z
        if fabs(diff) > DEFAULT_ABS_TOLERANCE:
            raise RuntimeError(
                f"compute_tensor_Green: solve_ivp returned sample points that differ from those requested (difference={diff} at i={i})"
            )

    return {
        "compute_time": supervisor.integration_time,
        "compute_steps": int(sol.nfev),
        "RHS_evaluations": supervisor.RHS_evaluations,
        "mean_RHS_time": supervisor.mean_RHS_time,
        "max_RHS_time": supervisor.max_RHS_time,
        "min_RHS_time": supervisor.min_RHS_time,
        "values": sampled_G,
        "solver_label": "solve_ivp+Radau-stepping0",
    }


class TensorGreenFunctionIntegration(DatastoreObject):
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
        cosmology: BaseCosmology,
        k: wavenumber_exit_time,
        atol: tolerance,
        rtol: tolerance,
        z_source: Optional[redshift] = None,
        z_sample: Optional[redshift_array] = None,
        label: Optional[str] = None,
        tags: Optional[List[store_tag]] = None,
    ):
        check_units(k.k, cosmology)
        self._solver_labels = solver_labels

        if payload is None:
            DatastoreObject.__init__(self, None)
            self._compute_time = None
            self._compute_steps = None
            self._RHS_evaluations = None
            self._mean_RHS_time = None
            self._max_RHS_time = None
            self._min_RHS_time = None

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

            self._solver = payload["solver"]

            self._z_sample = z_sample
            self._values = payload["values"]

        # check that all sample points are *later* than the specified source redshift
        if z_source is not None and self._z_sample is not None:
            z_init_float = float(z_source)
            for z in self._z_sample:
                z_float = float(z)
                if z_float > z_init_float:
                    raise ValueError(
                        f"Redshift sample point z={z_float} exceeds source redshift z={z_init_float}"
                    )

        # store parameters
        self._label = label
        self._tags = tags if tags is not None else []

        self._cosmology = cosmology

        self._k_exit = k
        self._z_source = z_source

        self._compute_ref = None

        self._atol = atol
        self._rtol = rtol

    @property
    def cosmology(self):
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
    def z_source(self):
        return self._z_source

    @property
    def z_sample(self):
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

        if self._z_source is None or self._z_sample is None:
            raise RuntimeError(
                "Object has not been configured correctly for a concrete calculation (z_source or z_sample is missing). It can only represent a query."
            )

        # replace label if specified
        if label is not None:
            self._label = label

        self._compute_ref = compute_tensor_Green.remote(
            self.cosmology,
            self.k,
            self.z_source,
            self.z_sample,
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
        self._RHS_evaluations = data["RHS_evaluations"]
        self._mean_RHS_time = data["mean_RHS_time"]
        self._max_RHS_time = data["max_RHS_time"]
        self._min_RHS_time = data["min_RHS_time"]

        values = data["values"]
        self._values = []

        for i in range(len(values)):
            # create new TensorGreenFunctionValue object
            self._values.append(
                TensorGreenFunctionValue(None, self._z_sample[i], values[i])
            )

        self._solver = self._solver_labels[data["solver_label"]]

        return True


class TensorGreenFunctionValue(DatastoreObject):
    """
    Encapsulates a single sampled value of the tensor Green's transfer functions.
    Parameters such as wavenumber k, source redshift z_source, etc., are held by the
    owning TensorGreenFunctionIntegration object
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
