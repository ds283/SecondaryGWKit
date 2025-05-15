from collections import namedtuple
from math import sqrt, fabs, log
from typing import Optional, List, Union

import ray
from ray import ObjectRef
from scipy.integrate import solve_ivp
from scipy.interpolate import make_interp_spline

from ComputeTargets.spline_wrappers import ZSplineWrapper
from CosmologyConcepts import redshift_array, redshift, wavenumber
from CosmologyModels import BaseCosmology
from Datastore import DatastoreObject
from MetadataConcepts import tolerance, store_tag
from Quadrature.integration_metadata import IntegrationSolver, IntegrationData
from Quadrature.supervisors.base import RHS_timer, IntegrationSupervisor
from Units.base import UnitsLike
from defaults import DEFAULT_ABS_TOLERANCE, DEFAULT_REL_TOLERANCE

A0_TAU_INDEX = 0
EXPECTED_SOL_LENGTH = 1

ModelFunctions = namedtuple(
    "ModelFunctions",
    [
        "Hubble",
        "epsilon",
        "d_epsilon_dz",
        "d2_epsilon_dz2",
        "wBackground",
        "wPerturbations",
        "tau",
        "d_lnH_dz",
        "d2_lnH_dz2",
        "d3_lnH_dz3",
        "d_wPerturbations_dz",
        "d2_wPerturbations_dz2",
    ],
)


@ray.remote
def compute_background(
    cosmology: BaseCosmology,
    z_sample: redshift_array,
    atol: float = DEFAULT_ABS_TOLERANCE,
    rtol: float = DEFAULT_REL_TOLERANCE,
) -> dict:
    z_init = float(z_sample.max)
    z_stop = float(z_sample.min)

    def RHS(z, state, supervisor) -> List[float]:
        with RHS_timer(supervisor) as timer:
            H = cosmology.Hubble(z)
            da0_tau_dz = -1.0 / H

            return [da0_tau_dz]

    with IntegrationSupervisor() as supervisor:
        rho_init = cosmology.rho(z_init)
        tau_init = (
            sqrt(3.0) * cosmology.units.PlanckMass / sqrt(rho_init) * (1.0 + z_init)
        )

        initial_state = [tau_init]

        sol = solve_ivp(
            RHS,
            method="RK45",
            t_span=(z_init, z_stop),
            y0=initial_state,
            t_eval=z_sample.as_float_list(),
            atol=atol,
            rtol=rtol,
            args=(supervisor,),
        )

    if not sol.success:
        raise RuntimeError(
            f'compute_background: integration did not terminate successfully (z_init={z_init:.5g}, z_stop={z_stop:.5g}, error at z={sol.t[-1]:.5g}, "{sol.message}")'
        )

    sampled_z = sol.t
    sampled_values = sol.y
    if len(sampled_values) != EXPECTED_SOL_LENGTH:
        raise RuntimeError(
            f"compute_background: solution does not have expected number of members (expected {EXPECTED_SOL_LENGTH}, found {len(sampled_values)}; length of sol.t={len(z_sample)})"
        )
    a0_tau_sample = sampled_values[A0_TAU_INDEX]

    returned_values = sampled_z.size
    expected_values = len(z_sample)

    if returned_values != expected_values:
        raise RuntimeError(
            f"compute_background: solve_ivp returned {returned_values} samples, but expected {expected_values}"
        )

    # validate that the samples of the solution correspond to the z-sample points that we specified.
    # This really should be true, but there is no harm in being defensive.
    for i in range(returned_values):
        diff = sampled_z[i] - z_sample[i].z
        if fabs(diff) > DEFAULT_ABS_TOLERANCE:
            raise RuntimeError(
                f"compute_background: solve_ivp returned sample points that differ from those requested (difference={diff} at i={i})"
            )

    # each BaseCosmology instance provides methods to evaluate H(z), rho(z), and the value of the equation of state
    # for background and perturbations
    H_sample = [cosmology.Hubble(z.z) for z in z_sample]
    rho_sample = [cosmology.rho(z.z) for z in z_sample]
    wBackground_sample = [cosmology.wBackground(z.z) for z in z_sample]
    wPerturbations_sample = [cosmology.wPerturbations(z.z) for z in z_sample]

    # further, each BaseCosmology instance may provide methods to evaluate the derivatives of H(z) and w(z), but if it doesn't,
    # we estimate these derivatives using a spline

    if hasattr(cosmology, "d_lnH_dz"):
        d_lnH_dz_sample = [cosmology.d_lnH_dz(z.z) for z in z_sample]
    else:
        lnH_data = [(log(1.0 + z.z), log(H)) for z, H in zip(z_sample, H_sample)]
        lnH_data.sort(key=lambda pair: pair[0])
        lnH_data_x, lnH_data_y = zip(*lnH_data)
        lnH_spline = make_interp_spline(lnH_data_x, lnH_data_y)
        raw_d_lnH_dz_spline = lnH_spline.derivative()
        d_lnH_dz_spline = ZSplineWrapper(
            raw_d_lnH_dz_spline,
            label="d_lnH_dz",
            min_z=z_sample.min.z,
            max_z=z_sample.max.z,
            log_z=True,
            deriv=True,
        )

        d_lnH_dz_sample = [d_lnH_dz_spline(z.z) for z in z_sample]

    if hasattr(cosmology, "d2_lnH_dz2"):
        d2_lnH_dz2_sample = [cosmology.d2_lnH_dz2(z.z) for z in z_sample]
    else:
        d_lnH_dz_data = [
            (log(1.0 + z.z), d_lnH_dz) for z, d_lnH_dz in zip(z_sample, d_lnH_dz_sample)
        ]
        d_lnH_dz_data.sort(key=lambda pair: pair[0])
        d_lnH_dz_data_x, d_lnH_dz_data_y = zip(*d_lnH_dz_data)
        raw_d_lnH_dz_spline = make_interp_spline(d_lnH_dz_data_x, d_lnH_dz_data_y)
        raw_d2_lnH_dz2_spline = raw_d_lnH_dz_spline.derivative()
        d2_lnH_dz2_spline = ZSplineWrapper(
            raw_d2_lnH_dz2_spline,
            label="d2_lnH_dz2",
            min_z=z_sample.min.z,
            max_z=z_sample.max.z,
            log_z=True,
            deriv=True,
        )

        d2_lnH_dz2_sample = [d2_lnH_dz2_spline(z.z) for z in z_sample]

    if hasattr(cosmology, "d3_lnH_dz3"):
        d3_lnH_dz3_sample = [cosmology.d3_lnH_dz3(z.z) for z in z_sample]
    else:
        d2_lnH_dz2_data = [
            (log(1.0 + z.z), d2_lnH_dz2)
            for z, d2_lnH_dz2 in zip(z_sample, d2_lnH_dz2_sample)
        ]
        d2_lnH_dz2_data.sort(key=lambda pair: pair[0])
        d2_lnH_dz2_data_x, d2_lnH_dz2_data_y = zip(*d2_lnH_dz2_data)
        raw_d2_lnH_dz2_spline = make_interp_spline(d2_lnH_dz2_data_x, d2_lnH_dz2_data_y)
        raw_d3_lnH_dz3_spline = raw_d2_lnH_dz2_spline.derivative()
        d3_lnH_dz3_spline = ZSplineWrapper(
            raw_d3_lnH_dz3_spline,
            label="d3_lnH_dz3",
            min_z=z_sample.min.z,
            max_z=z_sample.max.z,
            log_z=True,
            deriv=True,
        )

        d3_lnH_dz3_sample = [d3_lnH_dz3_spline(z.z) for z in z_sample]

    if hasattr(cosmology, "d_wPerturbations_dz"):
        d_wPerturbations_dz_sample = [
            cosmology.d_wPerturbations_dz(z.z) for z in z_sample
        ]
    else:
        wPerturbations_data = [
            (log(1.0 + z.z), wPerturbations)
            for z, wPerturbations in zip(z_sample, wPerturbations_sample)
        ]
        wPerturbations_data.sort(key=lambda pair: pair[0])
        wPerturbations_data_x, wPerturbations_data_y = zip(*wPerturbations_data)
        raw_wPerturbations_spline = make_interp_spline(
            wPerturbations_data_x, wPerturbations_data_y
        )
        raw_d_wPerturbations_dz_spline = raw_wPerturbations_spline.derivative()
        d_wPerturbations_dz_spline = ZSplineWrapper(
            raw_d_wPerturbations_dz_spline,
            label="d_wPerturbations_dz",
            min_z=z_sample.min.z,
            max_z=z_sample.max.z,
            log_z=True,
            deriv=True,
        )

        d_wPerturbations_dz_sample = [d_wPerturbations_dz_spline(z.z) for z in z_sample]

    if hasattr(cosmology, "d2_wPerturbations_dz2"):
        d2_wPerturbations_dz2_sample = [
            cosmology.d2_wPerturbations_dz2(z.z) for z in z_sample
        ]
    else:
        d_wPerturbations_dz_data = [
            (log(1.0 + z.z), d_wPerturbations_dz)
            for z, d_wPerturbations_dz in zip(z_sample, d_wPerturbations_dz_sample)
        ]
        d_wPerturbations_dz_data.sort(key=lambda pair: pair[0])
        d_wPerturbations_dz_data_x, d_wPerturbations_dz_data_y = zip(
            *d_wPerturbations_dz_data
        )
        raw_d_wPerturbations_dz_spline = make_interp_spline(
            d_wPerturbations_dz_data_x, d_wPerturbations_dz_data_y
        )
        raw_d2_wPerturbations_dz2_spline = raw_d_wPerturbations_dz_spline.derivative()
        d2_wPerturbations_dz2_spline = ZSplineWrapper(
            raw_d2_wPerturbations_dz2_spline,
            label="d2_wPerturbations_dz2",
            min_z=z_sample.min.z,
            max_z=z_sample.max.z,
            log_z=True,
            deriv=True,
        )

        d2_wPerturbations_dz2_sample = [
            d2_wPerturbations_dz2_spline(z.z) for z in z_sample
        ]

    return {
        "data": IntegrationData(
            compute_time=supervisor.integration_time,
            compute_steps=int(sol.nfev),
            RHS_evaluations=supervisor.RHS_evaluations,
            mean_RHS_time=supervisor.mean_RHS_time,
            max_RHS_time=supervisor.max_RHS_time,
            min_RHS_time=supervisor.min_RHS_time,
        ),
        "a0_tau_sample": a0_tau_sample,
        "H_sample": H_sample,
        "rho_sample": rho_sample,
        "wBackground_sample": wBackground_sample,
        "wPerturbations_sample": wPerturbations_sample,
        "d_lnH_dz_sample": d_lnH_dz_sample,
        "d2_lnH_dz2_sample": d2_lnH_dz2_sample,
        "d3_lnH_dz3_sample": d3_lnH_dz3_sample,
        "d_wPerturbations_dz_sample": d_wPerturbations_dz_sample,
        "d2_wPerturbations_dz2_sample": d2_wPerturbations_dz2_sample,
        "solver_label": "solve_ivp+RK45-stepping0",
    }


class BackgroundModel(DatastoreObject):
    """
    Encapsulates the time history of a cosmological model.
    This bakes-in all the quantities we need such as the conformal time \tau (for analytic
    approximations to the transfer functions and Green's functions).
    It also means we have an explicit record in the database of the values of H(z), w(z), etc.,
    that yielded a particular set of results
    """

    def __init__(
        self,
        payload,
        solver_labels: dict,
        cosmology: BaseCosmology,
        atol: tolerance,
        rtol: tolerance,
        z_sample: Optional[redshift_array] = None,
        label: Optional[str] = None,
        tags: Optional[List[store_tag]] = None,
    ):
        self._solver_labels = solver_labels
        self._z_sample = z_sample

        if payload is None:
            DatastoreObject.__init__(self, None)
            self._data = None
            self._solver = None
            self._values = None

        else:
            DatastoreObject.__init__(self, payload["store_id"])
            self._data: Optional[IntegrationData] = payload["data"]
            self._solver: Optional[IntegrationSolver] = payload["solver"]
            self._values: Optional[List[BackgroundModelValue]] = payload["values"]

        # store parameters
        self._label = label
        self._tags = tags if tags is not None else []

        self._cosmology = cosmology

        self._functions = None

        self._compute_ref = None

        self._atol = atol
        self._rtol = rtol

    @property
    def cosmology(self):
        return self._cosmology

    @property
    def label(self) -> str:
        return self._label

    @property
    def tags(self) -> List[store_tag]:
        return self._tags

    @property
    def z_sample(self):
        return self._z_sample

    def efolds_subh(self, k: wavenumber, z: Union[redshift, float]) -> float:
        if isinstance(z, redshift):
            z_float = z.z
        else:
            z_float = float(z)

        H = self.functions.Hubble(z_float)
        return log((1.0 + z_float) * k.k / H)

    @property
    def data(self) -> IntegrationData:
        if self.values is None:
            raise RuntimeError("values have not yet been populated")

        return self._data

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

    @property
    def functions(self) -> ModelFunctions:
        if self._values is None:
            raise RuntimeError("values has not yet been populated")

        if self._functions is None:
            self._create_functions()

        return self._functions

    def _create_functions(self):
        tau_data = [(log(1.0 + v.z.z), v.tau) for v in self.values]
        tau_data.sort(key=lambda pair: pair[0])

        tau_x_data, tau_y_data = zip(*tau_data)
        tau_spline = make_interp_spline(tau_x_data, tau_y_data)
        tau_func = ZSplineWrapper(
            tau_spline,
            label="tau",
            min_z=self.z_sample.min.z,
            max_z=self.z_sample.max.z,
            log_z=True,
        )

        if hasattr(self._cosmology, "d_lnH_dz"):
            d_lnH_dz_func = self._cosmology.d_lnH_dz
        else:
            d_lnH_dz_data = [(log(1.0 + v.z.z), v.d_lnH_dz) for v in self.values]
            d_lnH_dz_data.sort(key=lambda pair: pair[0])

            d_lnH_dz_x_data, d_lnH_dz_y_data = zip(*d_lnH_dz_data)
            d_lnH_dz_spline = make_interp_spline(d_lnH_dz_x_data, d_lnH_dz_y_data)
            d_lnH_dz_func = ZSplineWrapper(
                d_lnH_dz_spline,
                label="d_lnH_dz",
                min_z=self.z_sample.min.z,
                max_z=self.z_sample.max.z,
                log_z=True,
            )

        if hasattr(self._cosmology, "d2_lnH_dz2"):
            d2_lnH_dz2_func = self._cosmology.d2_lnH_dz2
        else:
            d2_lnH_dz2_data = [(log(1.0 + v.z.z), v.d2_lnH_dz2) for v in self.values]
            d2_lnH_dz2_data.sort(key=lambda pair: pair[0])

            d2_lnH_dz2_x_data, d2_lnH_dz2_y_data = zip(*d2_lnH_dz2_data)
            d2_lnH_dz2_spline = make_interp_spline(d2_lnH_dz2_x_data, d2_lnH_dz2_y_data)
            d2_lnH_dz2_func = ZSplineWrapper(
                d2_lnH_dz2_spline,
                label="d2_lnH_dz2",
                min_z=self.z_sample.min.z,
                max_z=self.z_sample.max.z,
                log_z=True,
            )

        if hasattr(self._cosmology, "d3_lnH_d32"):
            d3_lnH_dz3_func = self._cosmology.d3_lnH_d32
        else:
            d3_lnH_dz3_data = [(log(1.0 + v.z.z), v.d3_lnH_dz3) for v in self.values]
            d3_lnH_dz3_data.sort(key=lambda pair: pair[0])

            d3_lnH_dz3_x_data, d3_lnH_dz3_y_data = zip(*d3_lnH_dz3_data)
            d3_lnH_dz3_spline = make_interp_spline(d3_lnH_dz3_x_data, d3_lnH_dz3_y_data)
            d3_lnH_dz3_func = ZSplineWrapper(
                d3_lnH_dz3_spline,
                label="d3_lnH_dz3",
                min_z=self.z_sample.min.z,
                max_z=self.z_sample.max.z,
                log_z=True,
            )

        if hasattr(self._cosmology, "d_wPerturbations_dz"):
            d_wPerturbations_dz_func = self._cosmology.d_wPerturbations_dz
        else:
            d_wPerturbations_dz_data = [
                (log(1.0 + v.z.z), v.d_wPerturbations_dz) for v in self.values
            ]
            d_wPerturbations_dz_data.sort(key=lambda pair: pair[0])

            d_wPerturbations_dz_x_data, d_wPerturbations_dz_y_data = zip(
                *d_wPerturbations_dz_data
            )
            d_wPerturbations_dz_spline = make_interp_spline(
                d_wPerturbations_dz_x_data, d_wPerturbations_dz_y_data
            )
            d_wPerturbations_dz_func = ZSplineWrapper(
                d_wPerturbations_dz_spline,
                label="d_wPerturbations_dz",
                min_z=self.z_sample.min.z,
                max_z=self.z_sample.max.z,
                log_z=True,
            )

        if hasattr(self._cosmology, "d2_wPerturbations_dz2"):
            d2_wPerturbations_dz2_func = self._cosmology.d2_wPerturbations_dz2
        else:
            d2_wPerturbations_dz2_data = [
                (log(1.0 + v.z.z), v.d2_wPerturbations_dz2) for v in self.values
            ]
            d2_wPerturbations_dz2_data.sort(key=lambda pair: pair[0])

            d2_wPerturbations_dz2_x_data, d2_wPerturbations_dz2_y_data = zip(
                *d2_wPerturbations_dz2_data
            )
            d2_wPerturbations_dz2_spline = make_interp_spline(
                d2_wPerturbations_dz2_x_data, d2_wPerturbations_dz2_y_data
            )
            d2_wPerturbations_dz2_func = ZSplineWrapper(
                d2_wPerturbations_dz2_spline,
                label="d2_wPerturbations_dz2",
                min_z=self.z_sample.min.z,
                max_z=self.z_sample.max.z,
                log_z=True,
            )

        def epsilon(z: float) -> float:
            """
            Evaluate the conventional epsilon parameter eps = -dot(H)/H^2
            :param z: redshift of evaluation
            :return:
            """
            one_plus_z = 1.0 + z
            return one_plus_z * d_lnH_dz_func(z)

        def d_epsilon_dz(z: float) -> float:
            """
            Evaluate the z derivative of the epsilon parameter
            :param z:
            :return:
            """
            one_plus_z = 1.0 + z
            return d_lnH_dz_func(z) + one_plus_z * d2_lnH_dz2_func(z)

        def d2_epsilon_dz2(z: float) -> float:
            """
            Evaluate the 2nd z derivative of the epsilon parameter
            :param z:
            :return:
            """
            one_plus_z = 1.0 + z
            return 2.0 * d2_lnH_dz2_func(z) + one_plus_z * d3_lnH_dz3_func(z)

        self._functions = ModelFunctions(
            Hubble=self._cosmology.Hubble,
            epsilon=epsilon,
            d_epsilon_dz=d_epsilon_dz,
            d2_epsilon_dz2=d2_epsilon_dz2,
            wBackground=self._cosmology.wBackground,
            wPerturbations=self._cosmology.wPerturbations,
            tau=tau_func,
            d_lnH_dz=d_lnH_dz_func,
            d2_lnH_dz2=d2_lnH_dz2_func,
            d3_lnH_dz3=d3_lnH_dz3_func,
            d_wPerturbations_dz=d_wPerturbations_dz_func,
            d2_wPerturbations_dz2=d2_wPerturbations_dz2_func,
        )

    def compute(self, label: Optional[str] = None):
        if self._values is not None:
            raise RuntimeError("values has not yet been populated")

        if self._z_sample is None:
            raise RuntimeError(
                "Object has not been configured correctly for a concrete calculation (z_sample is missing). It can only represent a query."
            )

        # replace label if specified
        if label is not None:
            self._label = label

        self._compute_ref = compute_background.remote(
            self.cosmology,
            self._z_sample,
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

        self._data = data["data"]

        H_sample = data["H_sample"]
        wB_sample = data["wBackground_sample"]
        wP_sample = data["wPerturbations_sample"]
        rho_sample = data["rho_sample"]
        tau_sample = data["a0_tau_sample"]

        d_lnH_ds_sample = data["d_lnH_dz_sample"]
        d2_lnH_dz2_sample = data["d2_lnH_dz2_sample"]
        d3_lnH_dz3_sample = data["d3_lnH_dz3_sample"]

        d_wPerturbations_dz_sample = data["d_wPerturbations_dz_sample"]
        d2_wPerturbations_dz2_sample = data["d2_wPerturbations_dz2_sample"]

        self._values = []
        for i in range(len(H_sample)):
            self._values.append(
                BackgroundModelValue(
                    None,
                    self._z_sample[i],
                    Hubble=H_sample[i],
                    wBackground=wB_sample[i],
                    wPerturbations=wP_sample[i],
                    rho=rho_sample[i],
                    tau=tau_sample[i],
                    d_lnH_dz=d_lnH_ds_sample[i],
                    d2_lnH_dz2=d2_lnH_dz2_sample[i],
                    d3_lnH_dz3=d3_lnH_dz3_sample[i],
                    d_wPerturbations_dz=d_wPerturbations_dz_sample[i],
                    d2_wPerturbations_dz2=d2_wPerturbations_dz2_sample[i],
                )
            )

        self._solver = self._solver_labels[data["solver_label"]]

        return True


class BackgroundModelValue(DatastoreObject):
    def __init__(
        self,
        store_id: int,
        z: redshift,
        Hubble: float,
        wBackground: float,
        wPerturbations: float,
        rho: float,
        tau: float,
        d_lnH_dz: float,
        d2_lnH_dz2: Optional[float] = None,
        d3_lnH_dz3: Optional[float] = None,
        d_wPerturbations_dz: Optional[float] = None,
        d2_wPerturbations_dz2: Optional[float] = None,
    ):
        DatastoreObject.__init__(self, store_id)

        self._z = z

        self._Hubble: float = Hubble
        self._wBackground: float = wBackground
        self._wPerturbations: float = wPerturbations
        self._rho: float = rho
        self._tau: float = tau

        self._d_lnH_dz: float = d_lnH_dz
        self._d2_lnH_dz2: float = d2_lnH_dz2
        self._d3_lnH_dz3: float = d3_lnH_dz3

        self._d_wPerturbations_dz: float = d_wPerturbations_dz
        self._d2_wPerturbations_dz2: float = d2_wPerturbations_dz2

    @property
    def z(self) -> redshift:
        return self._z

    @property
    def Hubble(self) -> float:
        return self._Hubble

    @property
    def wBackground(self) -> float:
        return self._wBackground

    @property
    def wPerturbations(self) -> float:
        return self._wPerturbations

    @property
    def rho(self) -> float:
        return self._rho

    @property
    def tau(self) -> float:
        return self._tau

    @property
    def d_lnH_dz(self) -> float:
        return self._d_lnH_dz

    @property
    def d2_lnH_dz2(self) -> Optional[float]:
        return self._d2_lnH_dz2

    @property
    def d3_lnH_dz3(self) -> Optional[float]:
        return self._d3_lnH_dz3

    @property
    def d_wPerturbations_dz(self) -> Optional[float]:
        return self._d_wPerturbations_dz

    @property
    def d2_wPerturbations_dz2(self) -> Optional[float]:
        return self._d2_wPerturbations_dz2


class ModelProxy:
    def __init__(self, model: BackgroundModel):
        self._ref: ObjectRef = ray.put(model)

        self._store_id: int = model.store_id if model.available else None

        self._units: UnitsLike = model.cosmology.units
        self._cosmology: BaseCosmology = model.cosmology

    @property
    def store_id(self) -> int:
        return self._store_id

    @property
    def available(self) -> bool:
        return self._store_id is not None

    @property
    def units(self) -> UnitsLike:
        return self._units

    @property
    def cosmology(self) -> BaseCosmology:
        return self._cosmology

    def get(self) -> BackgroundModel:
        """
        The return value should only be held locally and not persisted, otherwise the entire
        BackgroundModel instance may be serialized when it is passed around by Ray.
        That would defeat the purpose of the proxy.
        :return:
        """
        return ray.get(self._ref)
