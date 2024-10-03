from typing import Optional, List

import ray
from math import log

from ComputeTargets import IntegrationSolver
from ComputeTargets.WKB_tensor_Green import WKB_omegaEff_sq
from ComputeTargets.analytic_tensor_Green import (
    compute_analytic_G,
    compute_analytic_Gprime,
)
from CosmologyConcepts import wavenumber_exit_time, redshift, redshift_array, wavenumber
from CosmologyModels import BaseCosmology
from Datastore import DatastoreObject
from MetadataConcepts import tolerance, store_tag
from defaults import DEFAULT_FLOAT_PRECISION
from utilities import check_units


class TensorGreenWKB(DatastoreObject):
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

        self._compute_ref = compute_tensor_Green_WKB.remote(
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
                "TensorGreenWKBIntegration: store() called, but no compute() is in progress"
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
        k_over_aH = (1.0 + self.z_source.z) * self.k.k / Hsource
        self._init_efolds_suph = -log(k_over_aH)

        theta_sample = data["G_sample"]
        a0_tau_sample = data["a0_tau_sample"]
        self._values = []

        tau_source = a0_tau_sample[0]
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

            # create new TensorGreenFunctionValue object
            self._values.append(
                TensorGreenWKBValue(
                    None,
                    current_z,
                    H,
                    theta_sample[i],
                    omega_WKB_sq=omega_WKB_sq,
                    analytic_G=analytic_G,
                    analytic_Gprime=analytic_Gprime,
                )
            )

        self._solver = self._solver_labels[data["solver_label"]]

        return True


class TensorGreenWKBValue(DatastoreObject):
    def __init__(
        self,
        store_id: int,
        z: redshift,
        H: float,
        theta: float,
        omega_WKB_sq: Optional[float] = None,
        analytic_G: Optional[float] = None,
        analytic_Gprime: Optional[float] = None,
    ):
        DatastoreObject.__init__(self, store_id)

        self._z = z
        self._H = H

        self._theta = theta
        self._omega_WKB_sq = omega_WKB_sq

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
    def omega_WKB_sq(self) -> Optional[float]:
        return self._omega_WKB_sq

    @property
    def H(self) -> float:
        return self._H

    @property
    def theta(self) -> float:
        return self._theta

    @property
    def analytic_G(self) -> Optional[float]:
        return self._analytic_G

    @property
    def analytic_Gprime(self) -> Optional[float]:
        return self._analytic_Gprime
