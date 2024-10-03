import time
from typing import Optional, List

import ray
from math import log, sqrt, fabs, cos, sin
from scipy.constants import epsilon_0
from scipy.integrate import solve_ivp

from ComputeTargets import IntegrationSolver
from ComputeTargets.WKB_tensor_Green import WKB_omegaEff_sq, WKB_d_ln_omegaEffPrime_dz
from ComputeTargets.analytic_Gk import (
    compute_analytic_G,
    compute_analytic_Gprime,
)
from ComputeTargets.integration_supervisor import (
    RHS_timer,
    IntegrationSupervisor,
    DEFAULT_UPDATE_INTERVAL,
)
from CosmologyConcepts import wavenumber_exit_time, redshift, redshift_array, wavenumber
from CosmologyModels import BaseCosmology
from Datastore import DatastoreObject
from MetadataConcepts import tolerance, store_tag
from defaults import (
    DEFAULT_FLOAT_PRECISION,
    DEFAULT_ABS_TOLERANCE,
    DEFAULT_REL_TOLERANCE,
)
from utilities import check_units, format_time

A0_TAU_INDEX = 0
THETA_INDEX = 1


class GkWKBSupervisor(IntegrationSupervisor):
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

        z_complete = self._z_source - current_z
        z_remain = self._z_range - z_complete
        percent_remain = 100.0 * (z_remain / self._z_range)
        print(
            f"** STATUS UPDATE #{update_number}: Integration for WKB Theta_k(z) for k = {self._k.k_inv_Mpc:.5g}/Mpc (store_id={self._k.store_id}) has been running for {format_time(since_start)} ({format_time(since_last_notify)} since last notification)"
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
def compute_Gk_WKB(
    cosmology: BaseCosmology,
    k: wavenumber_exit_time,
    z_source: redshift,
    z_sample: redshift_array,
    atol: float = DEFAULT_ABS_TOLERANCE,
    rtol: float = DEFAULT_REL_TOLERANCE,
) -> dict:
    k_wavenumber: wavenumber = k.k
    check_units(k_wavenumber, cosmology)

    k_float = k_wavenumber.k

    z_min = float(z_sample.min)

    def RHS(z, state, supervisor) -> List[float]:
        with RHS_timer(supervisor) as timer:
            a0_tau = state[A0_TAU_INDEX]
            theta = state[THETA_INDEX]

            if supervisor.notify_avalable:
                supervisor.message(z, f"current state: theta_WKB = {theta:.5g}")
                supervisor.reset_notify_time()

            omega_eff_sq = WKB_omegaEff_sq(cosmology, k_float, z)

            if omega_eff_sq < 0.0:
                raise ValueError(
                    f"omega_WKB^2 cannot be negative during WKB integration (omega_WKB^2={omega_eff_sq:.5g})"
                )

            omega_eff = sqrt(omega_eff_sq)
            dtheta_dz = omega_eff

            H = cosmology.Hubble(z)
            da0_tau_dz = -1.0 / H

            return [da0_tau_dz, dtheta_dz]

    with GkWKBSupervisor(k_wavenumber, z_source, z_sample.min) as supervisor:
        rho_init = cosmology.rho(z_source.z)
        tau_init = (
            sqrt(3.0) * cosmology.units.PlanckMass / sqrt(rho_init) * (1.0 + z_source.z)
        )

        initial_state = [tau_init, 0.0]

        sol = solve_ivp(
            RHS,
            method="RK45",
            t_span=(z_source.z, z_min),
            y0=initial_state,
            t_eval=z_sample.as_list(),
            atol=atol,
            rtol=rtol,
            args=(supervisor,),
        )

    if not sol.success:
        raise RuntimeError(
            f'compute_Gk_WKB: integration did not terminate successfully (k={k_wavenumber.k.k_inv_Mpc}/Mpc, z_source={z_source.z}, error at z={sol.t[-1]}, "{sol.message}")'
        )

    sampled_z = sol.t
    sampled_values = sol.y
    sampled_a0_tau = sampled_values[A0_TAU_INDEX]
    sampled_theta = sampled_values[THETA_INDEX]

    returned_values = sampled_z.size
    expected_values = len(z_sample)

    if returned_values != expected_values:
        raise RuntimeError(
            f"compute_Gk: solve_ivp returned {returned_values} samples, but expected {expected_values}"
        )

    # validate that the samples of the solution correspond to the z-sample points that we specified.
    # This really should be true, but there is no harm in being defensive.
    for i in range(returned_values):
        diff = sampled_z[i] - z_sample[i].z
        if fabs(diff) > DEFAULT_ABS_TOLERANCE:
            raise RuntimeError(
                f"compute_Gk_WKB: solve_ivp returned sample points that differ from those requested (difference={diff} at i={i})"
            )

    return {
        "compute_time": supervisor.integration_time,
        "compute_steps": int(sol.nfev),
        "RHS_evaluations": supervisor.RHS_evaluations,
        "mean_RHS_time": supervisor.mean_RHS_time,
        "max_RHS_time": supervisor.max_RHS_time,
        "min_RHS_time": supervisor.min_RHS_time,
        "a0_tau_sample": sampled_a0_tau,
        "theta": sampled_theta,
        "solver_label": "solve_ivp+RK45-stepping0",
    }


class GkWKBIntegration(DatastoreObject):
    """
    Encapsulates all sample points produced for a calculation of the WKB
    phase associated with the tensor Green's function
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
        G_init: Optional[float] = 0.0,
        Gprime_init: Optional[float] = 1.0,
        label: Optional[str] = None,
        tags: Optional[List[store_tag]] = None,
    ):
        k_wavenumber: wavenumber = k.k
        check_units(k_wavenumber, cosmology)

        self._solver_labels = solver_labels
        self._z_sample = z_sample

        self._G_init = G_init
        self._Gprime_init = Gprime_init

        if payload is None:
            DatastoreObject.__init__(self, None)
            self._compute_time = None
            self._compute_steps = None
            self._RHS_evaluations = None
            self._mean_RHS_time = None
            self._max_RHS_time = None
            self._min_RHS_time = None

            self._init_efolds_suph = None

            self._solver = None

            self._sin_coeff = None
            self._cos_coeff = None
            self._values = None

        else:
            DatastoreObject.__init__(self, payload["store_id"])
            self._compute_time = payload["compute_time"]
            self._compute_steps = payload["compute_steps"]
            self._RHS_evaluations = payload["RHS_evaluations"]
            self._mean_RHS_time = payload["mean_RHS_time"]
            self._max_RHS_time = payload["max_RHS_time"]
            self._min_RHS_time = payload["min_RHS_time"]

            self._init_efolds_subh = payload["init_efolds_subh"]

            self._solver = payload["solver"]

            self._sin_coeff = payload["sin_coeff"]
            self._cos_coeff = payload["cos_coeff"]
            self._values = payload["values"]

        if self._z_sample is not None:
            z_limit = k.z_exit_subh_e3
            z_source_float = float(z_source)
            for z in self._z_sample:
                # check that each response redshift is not too close to the horizon, or outside it, for this k-mode
                z_float = float(z)
                if z_float > z_limit - DEFAULT_FLOAT_PRECISION:
                    raise ValueError(
                        f"Specified response redshift z={z_float:.5g} is closer than 3-folds to horizon re-entry for wavenumber k={k_wavenumber:.5g}/Mpc"
                    )

                # also, check that each response redshift is later than the specified source redshift
                if z_float > z_source_float:
                    raise ValueError(
                        f"Redshift sample point z={z_float:.5g} exceeds source redshift z={z_source_float:.5g}"
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
    def init_efolds_subh(self) -> float:
        if self._init_efolds_subh is None:
            raise RuntimeError("init_efolds_subh has not yet been populated")

        return self._init_efolds_subh

    @property
    def sin_coeff(self) -> float:
        if self._sin_coeff is None:
            raise RuntimeError("sin_coeff has not yet been populated")

        return self._sin_coeff

    @property
    def cos_coeff(self) -> float:
        if self._cos_coeff is None:
            raise RuntimeError("cos_coeff has not yet been populated")

        return self._cos_coeff

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

        self._compute_ref = compute_Gk_WKB.remote(
            self.cosmology,
            self._k_exit,
            self.z_source,
            self.z_sample,
            atol=self._atol.tol,
            rtol=self._rtol.tol,
        )
        return self._compute_ref

    def store(self) -> Optional[bool]:
        if self._compute_ref is None:
            raise RuntimeError(
                "GkWKBIntegration: store() called, but no compute() is in progress"
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

        Hsource = self.cosmology.Hubble(self.z_source.z)
        epsSource = self.cosmology.eps(self.z_source.z)
        one_plus_z_source = 1.0 + self.z_source.z
        k_over_aH = one_plus_z_source * self.k.k / Hsource
        self._init_efolds_suph = -log(k_over_aH)

        theta_sample = data["G_sample"]
        a0_tau_sample = data["a0_tau_sample"]
        self._values = []

        tau_source = a0_tau_sample[0]
        omega_WKB_sq_source = WKB_omegaEff_sq(
            self._cosmology, self.k.k, self.z_source.z
        )
        if omega_WKB_sq_source < 0.0:
            raise ValueError(
                f"omega_WKB^2 must be non-negative at the initial time (omega_WKB^2={omega_WKB_sq_source:.5g})"
            )

        omega_WKB_source = sqrt(omega_WKB_sq_source)
        self._cos_coeff = omega_WKB_source * self._G_init

        d_ln_omega_WKB = WKB_d_ln_omegaEffPrime_dz(
            self._cosmology, self.k.k, self.z_source.z
        )
        self._sin_coeff = (
            self._Gprime_init
            + (self._G_init / 2.0) * (epsSource / one_plus_z_source + d_ln_omega_WKB)
        ) / omega_WKB_source

        # need to be aware that G_sample may not be as long as self._z_sample, if we are working in "stop" mode
        for i in range(len(theta_sample)):
            current_z = self._z_sample[i]
            current_z_float = current_z.z
            H = self._cosmology.Hubble(current_z_float)
            tau = a0_tau_sample[i]

            analytic_G = compute_analytic_G(
                self.k.k, 1.0 / 3.0, tau_source, tau, Hsource
            )
            analytic_Gprime = compute_analytic_Gprime(
                self.k.k, 1.0 / 3.0, tau_source, tau, Hsource, H
            )
            omega_WKB_sq = WKB_omegaEff_sq(self._cosmology, self.k.k, current_z_float)
            if omega_WKB_sq < 0.0:
                raise ValueError(
                    f"omega_WKB^2 must be non-negative at the initial time (omega_WKB^2={omega_WKB_sq:.5g})"
                )

            omega_WKB = sqrt(omega_WKB_sq)

            H_ratio = Hsource / H

            G_WKB = self._cos_coeff / omega_WKB * sqrt(H_ratio) * cos(
                theta_sample[i]
            ) + self._sin_coeff / omega_WKB * sqrt(H_ratio) * sin(theta_sample[i])

            # create new GkWKBValue object
            self._values.append(
                GkWKBValue(
                    None,
                    current_z,
                    Hsource / H,
                    theta_sample[i],
                    omega_WKB_sq=omega_WKB_sq,
                    G_WKB=G_WKB,
                    analytic_G=analytic_G,
                    analytic_Gprime=analytic_Gprime,
                )
            )

        self._solver = self._solver_labels[data["solver_label"]]

        return True


class GkWKBValue(DatastoreObject):
    def __init__(
        self,
        store_id: int,
        z: redshift,
        H_ratio: float,
        theta: float,
        omega_WKB_sq: Optional[float] = None,
        G_WKB: Optional[float] = None,
        analytic_G: Optional[float] = None,
        analytic_Gprime: Optional[float] = None,
    ):
        DatastoreObject.__init__(self, store_id)

        self._z = z
        self._H_ratio = H_ratio

        self._theta = theta
        self._omega_WKB_sq = omega_WKB_sq
        self._G_WKB = G_WKB

        self._analytic_G = analytic_G
        self._analytic_Gprime = analytic_Gprime

    def __float__(self):
        """
        Cast to float. Returns value of the transfer function
        :return:
        """
        return self.G

    @property
    def z(self) -> redshift:
        return self._z

    @property
    def H_ratio(self) -> float:
        return self._H_ratio

    @property
    def theta(self) -> float:
        return self._theta

    @property
    def omega_WKB_sq(self) -> Optional[float]:
        return self._omega_WKB_sq

    @property
    def G_WKB(self) -> Optional[float]:
        return self._G_WKB_sq

    @property
    def analytic_G(self) -> Optional[float]:
        return self._analytic_G

    @property
    def analytic_Gprime(self) -> Optional[float]:
        return self._analytic_Gprime
